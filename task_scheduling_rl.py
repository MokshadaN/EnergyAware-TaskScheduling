"""Energy-aware task scheduling with a DQN agent.

This module implements a minimal Deep Q-Network setup that follows the
algorithmic outline in the project notes:
- TaskSchedulingDQN model setup
- epsilon-greedy action selection
- one-step Q-learning update
- epsilon decay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import random

import torch
from torch import Tensor, nn, optim


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the DQN-based scheduling agent."""

    n_input: int
    n_actions: int
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class TaskSchedulingDQN(nn.Module):
    """Simple 2-layer MLP used as the Q-network."""

    def __init__(self, n_input: int, n_actions: int, alpha: float) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_actions = n_actions

        self.layer1 = nn.Linear(n_input, n_input)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(n_input, n_actions)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.relu(self.layer1(state))
        return self.layer2(hidden)


class AgentClass:
    """DQN agent with epsilon-greedy policy for task scheduling."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay

        self.action_space = list(range(config.n_actions))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = TaskSchedulingDQN(
            n_input=config.n_input,
            n_actions=config.n_actions,
            alpha=config.learning_rate,
        ).to(self.device)

    def choose_action(self, state: Sequence[float] | Tensor) -> int:
        """Choose an action via epsilon-greedy exploration/exploitation."""
        if random.random() > self.epsilon:
            state_t = self._to_state_tensor(state)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
            return action

        return random.choice(self.action_space)

    def learn(
        self,
        state: Sequence[float] | Tensor,
        action: int,
        reward: float,
        next_state: Sequence[float] | Tensor,
    ) -> float:
        """Perform one DQN update step and return scalar training loss."""
        state_t = self._to_state_tensor(state)
        next_state_t = self._to_state_tensor(next_state)
        action_t = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        q_pred = self.q_net(state_t).gather(1, action_t).squeeze(1)

        with torch.no_grad():
            q_next = self.q_net(next_state_t).max(dim=1).values
            q_target = reward_t + self.gamma * q_next

        loss = self.q_net.loss_fn(q_pred, q_target)

        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

        self.update_epsilon()
        return float(loss.item())

    def update_epsilon(self) -> None:
        """Decay epsilon until epsilon_min is reached."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _to_state_tensor(self, state: Sequence[float] | Tensor) -> Tensor:
        state_t = (
            state.clone().detach() if isinstance(state, torch.Tensor) else torch.tensor(state)
        )
        state_t = state_t.to(dtype=torch.float32, device=self.device)

        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        return state_t
