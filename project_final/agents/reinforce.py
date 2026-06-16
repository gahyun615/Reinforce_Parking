"""
REINFORCE Agent (Policy Gradient baseline)
- Continuous action space (steering, throttle)
- Monte-Carlo policy gradient with return normalization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agents.checkpoint_utils import validate_obs_dim


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_dist(self, states: torch.Tensor) -> Normal:
        features = self.backbone(states)
        mu = torch.tanh(self.mu_head(features))
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


class EpisodeBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def push(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class REINFORCEAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.005,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.policy = PolicyNet(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = EpisodeBuffer()
        self._step = 0

    def select_action(self, state: np.ndarray, training: bool = True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_dist(state_t)

        action_t = dist.sample() if training else dist.mean
        action_t = action_t.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action_t).sum(dim=-1)

        action = action_t.squeeze(0).cpu().numpy()
        return action, float(log_prob.item()), None

    def store(self, state, action, reward, log_prob, value, done):
        # Keep interface compatible with existing on-policy trainer.
        del log_prob, value
        self.buffer.push(state, action, reward, done)

    def should_update(self) -> bool:
        # REINFORCE updates once per episode.
        return len(self.buffer) > 0 and bool(self.buffer.dones[-1])

    def update(self, last_state: np.ndarray, last_done: bool) -> dict | None:
        del last_state, last_done
        if len(self.buffer) == 0 or not self.buffer.dones[-1]:
            return None

        self._step += 1

        returns = []
        ret = 0.0
        for r in reversed(self.buffer.rewards):
            ret = r + self.gamma * ret
            returns.insert(0, ret)

        states_t = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        dist = self.policy.get_dist(states_t)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        loss = -(log_probs * returns_t).mean() - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.clear()
        return {
            "loss": float(loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str):
        torch.save({
            "obs_dim": self.obs_dim,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        validate_obs_dim(ckpt, self.obs_dim, "reinforce")
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt["step"]
