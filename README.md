# EnergyAware-TaskScheduling

> An enhanced implementation of **Energy-Aware Reinforcement Learning for Cloud Task Scheduling**, inspired by the IEEE paper *Designing Energy-Aware Scheduling and Task Allocation Algorithms for Online Reinforcement Learning Applications in Cloud Environments*.

## Overview

Cloud data centers consume enormous amounts of energy while executing large-scale AI and machine learning workloads. Traditional scheduling algorithms such as FCFS, Round Robin, and static heuristics fail to adapt to changing workloads, resulting in poor resource utilization, increased power consumption, and SLA violations.

This project presents a **next-generation implementation** of the proposed RL-based scheduler by redesigning the architecture for modularity, scalability, reproducibility, and production-readiness.

Instead of reproducing the paper, this repository extends its ideas with cleaner abstractions, configurable environments, improved reward engineering, richer evaluation metrics, and support for large-scale experimentation.

---

## Paper Reference

This project is based on:

**Designing Energy-Aware Scheduling and Task Allocation Algorithms for Online Reinforcement Learning Applications in Cloud Environments**

IEEE Transactions on Computational Social Systems, 2025.

---

# Problem Statement

Given

- thousands of incoming tasks
- heterogeneous cloud resources
- fluctuating workloads
- varying CPU, RAM, Memory and Disk utilization

the scheduler must decide

- Which VM should execute the task?
- When should it execute?
- How can overall energy consumption be minimized?
- How can QoS and throughput be maintained simultaneously?

The problem is modeled as a **Sequential Decision Making** problem and solved using **Deep Reinforcement Learning**.

---

# Proposed Architecture

```
                  Cloud Environment
                         │
                         ▼
               Resource Monitoring Layer
                         │
         ┌───────────────┴───────────────┐
         │                               │
   Task Features                  Resource Features
         │                               │
         └───────────────┬───────────────┘
                         ▼
                  RL Scheduling Agent
                  (Deep Q Network)
                         │
              Action: Assign Task → VM
                         │
                         ▼
               Execute & Collect Metrics
                         │
      Reward ← Energy + Utilization + QoS
                         │
                         ▼
                  Experience Replay
                         │
                         ▼
                  Network Update
```

---

# Key Features

- Deep Q-Network based scheduler
- Online Reinforcement Learning
- Energy-aware task allocation
- Dynamic resource scheduling
- Multi-objective optimization
- Adaptive reward function
- Configurable cloud simulation
- Real-time scheduling decisions
- Performance benchmarking
- Visualization of scheduling metrics

---

# Improvements over the Original Paper

Unlike the original implementation, this repository focuses on making the scheduler practical, extensible, and easier to experiment with.

### 1. Modular Architecture

The original work tightly couples scheduling logic with the learning pipeline.

This implementation separates

- Environment
- Agent
- Scheduler
- Reward Function
- Metrics
- Configuration
- Evaluation

making future research significantly easier.

---

### 2. Configurable Reward Engineering

Instead of fixed reward coefficients, reward weights are configurable.

This allows optimization for different objectives including

- Energy
- Throughput
- Response Time
- Resource Utilization
- SLA Violations

---

### 3. Improved State Representation

The scheduler observes a richer state including

- CPU Utilization
- RAM Utilization
- Memory Usage
- Disk Usage
- Queue Length
- VM Availability
- Task Priority
- Estimated Execution Time

instead of relying only on the limited metrics discussed in the paper.

---

### 4. Better Exploration Strategy

The paper uses a standard ε-Greedy policy.

This implementation supports

- Adaptive ε decay
- Dynamic exploration scheduling
- Configurable exploration parameters

allowing faster convergence under varying workloads.

---

### 5. Better Evaluation Pipeline

Instead of evaluating only energy consumption, this project tracks

- Energy Consumption
- Power Usage
- Throughput
- Resource Utilization
- Average Waiting Time
- Average Response Time
- VM Utilization
- Task Completion Rate
- SLA Violations

---

### 6. Extensible Scheduler

The scheduling framework can easily integrate

- Double DQN
- Dueling DQN
- PPO
- A2C
- Actor-Critic
- Multi-Agent RL

without changing the simulator.

---

### 7. Reproducible Experiments

The project supports

- configurable workloads
- deterministic random seeds
- experiment logging
- metric visualization

allowing consistent benchmarking across algorithms.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- OpenAI Gym-style Environment
- Reinforcement Learning

---

# Evaluation Metrics

The scheduler is evaluated using

- Total Energy Consumption
- Average CPU Utilization
- Memory Utilization
- Response Time
- Throughput
- Power Consumption
- Task Completion Ratio
- Average Waiting Time
- Makespan
- Resource Utilization
- Reward Convergence

---

# Future Improvements

- Double DQN
- PPO-based Scheduler
- Multi-Agent Reinforcement Learning
- Kubernetes Cluster Scheduling
- CloudSim Integration
- Ray RLlib Support
- Distributed Training
- Real-world Kubernetes Benchmarking

---

# Results

The proposed implementation aims to achieve

- Lower energy consumption
- Better resource utilization
- Reduced response time
- Higher throughput
- Improved scalability
- Stable RL convergence

while remaining modular enough for future research.

---


# License

MIT License
