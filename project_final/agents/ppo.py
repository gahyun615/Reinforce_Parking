"""
PPO Agent (Our Solution - Advanced Actor-Critic)
- Continuous action space (steering, throttle)
- Clipped surrogate objective to prevent destructive updates
- Multiple epochs of minibatch updates per rollout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ------------------------------------------------------------------
# Actor-Critic Network (shared backbone)
# ------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        self.mu_head  = nn.Linear(hidden_dim, action_dim)
        self.log_std  = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        mu       = torch.tanh(self.mu_head(features))
        std      = self.log_std.exp().expand_as(mu)
        value    = self.value_head(features).squeeze(-1)
        return mu, std, value

    def get_dist(self, x: torch.Tensor) -> Normal:
        mu, std, _ = self.forward(x)
        return Normal(mu, std)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Returns log_probs, values, entropy for given states and actions."""
        mu, std, values = self.forward(states)
        dist      = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1).mean()
        return log_probs, values, entropy


# ------------------------------------------------------------------
# Rollout Buffer
# ------------------------------------------------------------------

class PPORolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def get_tensors(self, device):
        return (
            torch.tensor(np.array(self.states),    dtype=torch.float32, device=device),
            torch.tensor(np.array(self.actions),   dtype=torch.float32, device=device),
            torch.tensor(self.log_probs,           dtype=torch.float32, device=device),
            torch.tensor(self.values,              dtype=torch.float32, device=device),
            torch.tensor(self.dones,               dtype=torch.float32, device=device),
        )


# ------------------------------------------------------------------
# PPO Agent
# ------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim:       int,
        action_dim:    int,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,   # GAE smoothing factor
        clip_eps:      float = 0.2,    # PPO clipping epsilon
        entropy_coef:  float = 0.01,
        value_coef:    float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps:       int   = 2048,   # rollout length before update
        n_epochs:      int   = 10,     # number of update epochs per rollout
        batch_size:    int   = 64,     # minibatch size
        device:        str   = "cpu",
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.device        = torch.device(device)

        self.ac        = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        self.buffer = PPORolloutBuffer()
        self._step  = 0

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mu, std, value = self.ac(state_t)

        dist = Normal(mu, std)

        if training:
            action_t = dist.sample()
        else:
            action_t = mu  # deterministic at eval

        action_t  = action_t.clamp(-1.0, 1.0)
        log_prob  = dist.log_prob(action_t).sum(dim=-1)

        action   = action_t.squeeze(0).cpu().numpy()
        log_prob = log_prob.squeeze(0).item()
        value    = value.squeeze(0).item()

        return action, log_prob, value

    def store(self, state, action, reward, log_prob, value, done):
        self.buffer.push(state, action, reward, log_prob, value, done)

    def should_update(self) -> bool:
        return len(self.buffer) >= self.n_steps

    def update(self, last_state: np.ndarray, last_done: bool) -> dict:
        """
        PPO update:
        1. Compute GAE advantages
        2. Run n_epochs of minibatch updates with clipped objective
        """
        self._step += 1

        # Bootstrap last value
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.ac(last_state_t)[2].item() if not last_done else 0.0

        # Compute GAE advantages and returns
        advantages, returns = self._compute_gae(last_value)

        # Get stored tensors
        states_t, actions_t, old_log_probs_t, _, _ = self.buffer.get_tensors(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(self.buffer)
        metrics_history = []

        for _ in range(self.n_epochs):
            # Random minibatch indices
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]

                mb_states      = states_t[idx]
                mb_actions     = actions_t[idx]
                mb_old_lp      = old_log_probs_t[idx]
                mb_advantages  = advantages_t[idx]
                mb_returns     = returns_t[idx]

                # Evaluate current policy
                new_log_probs, values, entropy = self.ac.evaluate(mb_states, mb_actions)

                # PPO clipped ratio
                ratio       = (new_log_probs - mb_old_lp).exp()
                surr1       = ratio * mb_advantages
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                actor_loss  = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = nn.functional.smooth_l1_loss(values, mb_returns)

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics_history.append({
                    "loss":        loss.item(),
                    "actor_loss":  actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy":     entropy.item(),
                })

        self.buffer.clear()

        # Return averaged metrics
        return {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0]}

    def _compute_gae(self, last_value: float):
        """Generalized Advantage Estimation (GAE-Lambda)."""
        rewards = self.buffer.rewards
        values  = self.buffer.values
        dones   = self.buffer.dones
        n       = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else last_value
            next_done  = dones[t]
            delta      = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae        = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]

        return advantages, returns

    def save(self, path: str):
        torch.save({
            "ac":        self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step":      self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt["step"]
