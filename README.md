# EnergyAware-TaskScheduling

This repository now includes a reference implementation of an energy-aware
reinforcement learning scheduler using a Deep Q-Network (DQN).

## Implemented components

- **Model initialization** via `TaskSchedulingDQN` with:
  - configurable input/action dimensions,
  - Adam optimizer,
  - MSE loss.
- **Agent initialization** via `AgentConfig` and `AgentClass` with:
  - discount factor (`gamma`),
  - epsilon-greedy exploration parameters (`epsilon`, `epsilon_min`,
    `epsilon_decay`),
  - explicit action space.
- **Action selection**:
  - exploit by selecting argmax Q-value with probability `1 - epsilon`,
  - explore by selecting a random action with probability `epsilon`.
- **Learning step**:
  - tensor conversion and device placement,
  - Q prediction/target computation,
  - MSE loss optimization with backpropagation.
- **Epsilon decay** after each learning step.

## Quick start

```python
from task_scheduling_rl import AgentClass, AgentConfig

config = AgentConfig(n_input=8, n_actions=4, learning_rate=1e-3)
agent = AgentClass(config)

state = [0.2] * 8
next_state = [0.3] * 8
action = agent.choose_action(state)
loss = agent.learn(state, action, reward=1.0, next_state=next_state)
```
