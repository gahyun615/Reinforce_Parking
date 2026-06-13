"""
DQN Agent (Baseline 1 - Value-based RL)
- Discretized action space for continuous parking problem
- Experience replay + target network
- Epsilon-greedy exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ------------------------------------------------------------------
# Q-Network
# ------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------
# DQN Agent
# ------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        obs_dim:       int,
        n_actions:     int,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.05,
        epsilon_decay: int   = 50_000,   # steps over which epsilon decays
        batch_size:    int   = 128,
        buffer_size:   int   = 100_000,
        target_update: int   = 500,      # steps between target net sync
        device:        str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        self.q_net     = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self._step = 0

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            # Linear epsilon decay
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_end + (1.0 - self.epsilon_end)
                * (1 - self._step / self.epsilon_decay),
            )
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        self._step += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Sync target network
        if self._step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_net":     self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step":      self._step,
            "epsilon":   self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step   = ckpt["step"]
        self.epsilon = ckpt["epsilon"]
