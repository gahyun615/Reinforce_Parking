"""
SAC Agent (Soft Actor-Critic)
- Continuous action space with squashed Gaussian policy
- Twin Q-networks, target networks, automatic entropy tuning
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from agents.checkpoint_utils import validate_obs_dim


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, state: torch.Tensor):
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = SquashedGaussianPolicy(obs_dim, action_dim).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=self.device
            )

        self.buffer = ReplayBuffer(buffer_size)
        self._step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, training: bool = True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if training:
                action_t, _ = self.actor.sample(state_t)
            else:
                action_t = self.actor.deterministic_action(state_t)
        action = action_t.squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return action, None, None

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, online: nn.Module, target: nn.Module):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def update(self) -> dict | None:
        if len(self.buffer) < self.batch_size:
            return None

        self._step += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states_t)
            q1_next = self.q1_target(next_states_t, next_actions)
            q2_next = self.q2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        q1_val = self.q1(states_t, actions_t)
        q2_val = self.q2(states_t, actions_t)
        q1_loss = F.mse_loss(q1_val, target_q)
        q2_loss = F.mse_loss(q2_val, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_actions, log_probs = self.actor.sample(states_t)
        q1_pi = self.q1(states_t, new_actions)
        q2_pi = self.q2(states_t, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_val = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "loss": float((q1_loss + q2_loss + actor_loss).item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": alpha_loss_val,
        }

    def save(self, path: str):
        payload = {
            "obs_dim": self.obs_dim,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "step": self._step,
        }
        if self.auto_entropy:
            payload["alpha_opt"] = self.alpha_opt.state_dict()
        torch.save(payload, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        validate_obs_dim(ckpt, self.obs_dim, "sac")
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])
        if self.auto_entropy and "alpha_opt" in ckpt:
            self.log_alpha = nn.Parameter(ckpt["log_alpha"].clone().to(self.device))
            self.alpha_opt = optim.Adam([self.log_alpha], lr=self.actor_opt.param_groups[0]["lr"])
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        else:
            self.log_alpha = ckpt["log_alpha"].to(self.device)
        self._step = ckpt["step"]
