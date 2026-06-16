"""
A2C Agent (Baseline 2 - Policy Gradient / Actor-Critic)
- Continuous action space (steering, throttle)
- Synchronous Advantage Actor-Critic
- Gaussian policy with learnable std
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agents.checkpoint_utils import validate_obs_dim


# ------------------------------------------------------------------
# Actor Network (Policy)
# ------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head  = nn.Linear(hidden_dim, action_dim)
        # Log std as learnable parameter (shared across states)
        self.log_std  = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        mu       = torch.tanh(self.mu_head(features))   # in (-1, 1)
        std      = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_dist(self, x: torch.Tensor) -> Normal:
        mu, std = self.forward(x)
        return Normal(mu, std)


# ------------------------------------------------------------------
# Critic Network (Value function)
# ------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# Rollout Buffer (on-policy, cleared after each update)
# ------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.log_probs   = []
        self.values      = []
        self.dones       = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


# ------------------------------------------------------------------
# A2C Agent
# ------------------------------------------------------------------

class A2CAgent:
    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        entropy_coef: float = 0.01,   # encourages exploration
        value_coef:   float = 0.5,    # weight for critic loss
        max_grad_norm: float = 0.5,
        n_steps:      int   = 128,    # steps to collect before each update
        device:       str   = "cpu",
    ):
        self.obs_dim = obs_dim
        self.gamma         = gamma
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps       = n_steps
        self.device        = torch.device(device)

        self.actor  = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.buffer = RolloutBuffer()
        self._step  = 0

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True):
        """
        Returns:
            action    : np.ndarray (action_dim,) clipped to [-1, 1]
            log_prob  : float (None if not training)
            value     : float (None if not training)
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist  = self.actor.get_dist(state_t)
            value = self.critic(state_t)

        if training:
            action_t = dist.sample()
        else:
            action_t = dist.mean   # deterministic at eval

        action_t  = action_t.clamp(-1.0, 1.0)
        log_prob  = dist.log_prob(action_t).sum(dim=-1)

        action   = action_t.squeeze(0).cpu().numpy()
        log_prob = log_prob.squeeze(0).item()
        value    = value.squeeze(0).item()

        return action, log_prob, value

    def store(self, state, action, reward, log_prob, value, done):
        self.buffer.push(state, action, reward, log_prob, value, done)

    def update(self, last_state: np.ndarray, last_done: bool) -> dict | None:
        """
        Compute advantages using n-step returns and update both networks.
        Called when buffer has n_steps transitions.
        """
        if len(self.buffer) < self.n_steps:
            return None

        self._step += 1

        # Bootstrap value for last state
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.critic(last_state_t).item() if not last_done else 0.0

        # Compute discounted returns (n-step)
        returns = []
        R = last_value
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            R = reward + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)

        # Convert to tensors
        states_t    = torch.tensor(np.array(self.buffer.states),   dtype=torch.float32, device=self.device)
        actions_t   = torch.tensor(np.array(self.buffer.actions),  dtype=torch.float32, device=self.device)
        returns_t   = torch.tensor(returns,                         dtype=torch.float32, device=self.device)
        old_lp_t    = torch.tensor(self.buffer.log_probs,          dtype=torch.float32, device=self.device)

        # Critic forward
        values_t = self.critic(states_t)

        # Advantage = Return - Baseline(V)
        advantages_t = returns_t - values_t.detach()
        # Normalize advantages for stability
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Actor forward (recompute log_probs for current policy)
        dist     = self.actor.get_dist(states_t)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1).mean()

        # Losses
        actor_loss  = -(log_probs * advantages_t).mean()
        critic_loss = nn.functional.smooth_l1_loss(values_t, returns_t)
        loss        = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        self.buffer.clear()

        return {
            "loss":        loss.item(),
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropy.item(),
        }

    def should_update(self) -> bool:
        return len(self.buffer) >= self.n_steps

    def save(self, path: str):
        torch.save({
            "obs_dim": self.obs_dim,
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step":      self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        validate_obs_dim(ckpt, self.obs_dim, "a2c")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt["step"]
