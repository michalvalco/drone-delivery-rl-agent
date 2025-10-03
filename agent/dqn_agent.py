# dqn_agent.py
"""
Minimal DQN agent with:
- 2x128 MLP
- ε-greedy policy
- SmoothL1Loss (Huber) for stability
- Target network hard updates
- Gradient clipping

Author: Michal Valčo
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ):
        self.device = torch.device(device)
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q = QNetwork(state_dim, action_dim).to(self.device)
        self.hard_update()

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        self.gamma = gamma
        self.action_dim = action_dim
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.grad_clip = grad_clip

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(s)  # [1, A]
            return int(torch.argmax(qvals, dim=1).item())

    def update(self, batch: list[Transition]) -> float:
        # unpack batch
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        # current Q(s,a)
        q_sa = self.q(states).gather(1, actions)  # [B,1]

        # target: r + gamma * max_a' Q_target(s', a') * (1-done)
        with torch.no_grad():
            next_q = self.target_q(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.criterion(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def hard_update(self):
        self.target_q.load_state_dict(self.q.state_dict())

    # -------- IO --------
    def save(self, path: str):
        torch.save(
            {
                "q_state_dict": self.q.state_dict(),
                "target_state_dict": self.target_q.state_dict(),
                "epsilon": self.epsilon,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.q.load_state_dict(data["q_state_dict"])
        self.target_q.load_state_dict(data.get("target_state_dict", data["q_state_dict"]))
        self.epsilon = float(data.get("epsilon", 0.0))
