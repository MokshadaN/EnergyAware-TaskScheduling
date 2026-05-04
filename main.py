# corrected_energy_aware_dqn.py

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================
# TASK & VM
# =========================

class Task:
    def __init__(self, tid):
        self.id = tid
        self.resource = random.uniform(1, 5)
        self.deadline = random.uniform(10, 20)
        self.exec_time = random.uniform(1, 5)


class VM:
    def __init__(self, vid):
        self.id = vid
        self.capacity = random.uniform(5, 10)
        self.utilization = 0
        self.energy_rate = random.uniform(0.3, 0.7)
        self.available_time = 0


# =========================
# ENVIRONMENT
# =========================

class Env:
    def __init__(self, n_vms=5):
        self.vms = [VM(i) for i in range(n_vms)]
        self.reset()

    def reset(self):
        self.time = 0
        self.tasks = [Task(i) for i in range(20)]
        return self.get_state()

    def get_state(self):
        U = np.mean([1 if vm.available_time <= self.time else 0 for vm in self.vms])
        M = np.mean([vm.capacity - vm.utilization for vm in self.vms])
        D = np.mean([vm.utilization for vm in self.vms])
        C = D
        R = M
        return np.array([U, M, D, C, R], dtype=np.float32)

    def step(self, action):
        if not self.tasks:
            return self.get_state(), 0, 0, 0, True

        task = self.tasks.pop(0)
        vm = self.vms[action]

        success = False
        energy = 0
        delay = 0

        if vm.utilization + task.resource <= vm.capacity:
            exec_time = task.exec_time

            vm.utilization += task.resource
            vm.available_time = self.time + exec_time

            energy = vm.energy_rate * vm.utilization * exec_time

            delay = max(0, vm.available_time - task.deadline)
            delay = min(delay, 50)  # 🔥 CLIPPED

            success = True
        else:
            delay = 50  # penalty

        # cooldown
        for v in self.vms:
            v.utilization *= 0.95

        next_state = self.get_state()
        done = len(self.tasks) == 0

        return next_state, success, energy, delay, done


# =========================
# DQN
# =========================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + 2, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state, action, dq1, dq2):
        x = torch.cat([state, action, dq1, dq2], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# AGENT
# =========================

class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=3000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # 🔥 FIXED
        self.epsilon_min = 0.01

        self.action_dim = action_dim

        self.wU, self.wM, self.wD = 0.8, 0.6, 0.6
        self.wC, self.wR = 0.7, 0.7

    def one_hot(self, a):
        vec = np.zeros(self.action_dim)
        vec[a] = 1
        return vec

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0)
        dq1 = torch.zeros((1,1))
        dq2 = torch.zeros((1,1))

        q_values = []
        for a in range(self.action_dim):
            a_vec = torch.FloatTensor(self.one_hot(a)).unsqueeze(0)
            q = self.model(state, a_vec, dq1, dq2)
            q_values.append(q[0][a].item())

        return np.argmax(q_values)

    def compute_stage_rewards(self, state):
        U, M, D, C, R = state
        R1 = self.wU*U + self.wM*M + self.wD*D
        R2 = self.wC*C + self.wR*R
        return R1, R2

    def remember(self, s, a, r, s2, done, dq1, dq2):
        self.memory.append((s, a, r, s2, done, dq1, dq2))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, s2, done, dq1, dq2 in batch:
            s = torch.FloatTensor(s).unsqueeze(0)
            s2 = torch.FloatTensor(s2).unsqueeze(0)
            dq1 = torch.FloatTensor([[dq1]])
            dq2 = torch.FloatTensor([[dq2]])

            a_vec = torch.FloatTensor(self.one_hot(a)).unsqueeze(0)

            target = r
            if not done:
                next_qs = []
                for na in range(self.action_dim):
                    na_vec = torch.FloatTensor(self.one_hot(na)).unsqueeze(0)
                    q = self.target(s2, na_vec, dq1, dq2)
                    next_qs.append(q[0][na].item())
                target += self.gamma * max(next_qs)

            pred = self.model(s, a_vec, dq1, dq2)[0][a]
            loss = (pred - target) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# TRAIN
# =========================

def train():
    env = Env()
    agent = Agent(5, len(env.vms))

    episodes = 100

    energy_log, delay_log, util_log = [], [], []

    for ep in range(episodes):
        state = env.reset()

        ep_energy, ep_delay, ep_util = 0, 0, 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, success, energy, delay, done = env.step(action)

            R1, R2 = agent.compute_stage_rewards(state)

            dq1, dq2 = R1, R2

            # 🔥 NORMALIZED REWARD
            energy_norm = energy / 100
            delay_norm = delay / 50

            reward = (
                0.5 * R1 +
                0.5 * R2 -
                0.2 * energy_norm -
                0.4 * delay_norm -
                1.0 * (not success)
            )

            if success:
                reward += 0.5  # encourage scheduling

            agent.remember(state, action, reward, next_state, done, dq1, dq2)
            agent.replay()

            ep_energy += energy
            ep_delay += delay
            util = np.mean([vm.utilization / vm.capacity for vm in env.vms])
            ep_util += util

            state = next_state
            steps += 1

            if done:
                break

        energy_log.append(ep_energy)
        delay_log.append(ep_delay)
        util_log.append(ep_util / steps)

        print(f"Ep {ep+1} | Energy {ep_energy:.2f} | Delay {ep_delay:.2f} | Util {util_log[-1]:.2f}")

        # 🔥 TARGET UPDATE
        if ep % 10 == 0:
            agent.target.load_state_dict(agent.model.state_dict())

    # =========================
    # PLOTS
    # =========================

    plt.figure()
    plt.plot(energy_log)
    plt.title("Energy Consumption")
    plt.xlabel("Episodes")
    plt.ylabel("Energy")
    plt.show()

    plt.figure()
    plt.plot(delay_log)
    plt.title("Task Delay")
    plt.xlabel("Episodes")
    plt.ylabel("Delay")
    plt.show()

    plt.figure()
    plt.plot(util_log)
    plt.title("Resource Utilization")
    plt.xlabel("Episodes")
    plt.ylabel("Utilization")
    plt.show()


if __name__ == "__main__":
    train()