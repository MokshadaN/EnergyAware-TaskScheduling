import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple

class DQNNetwork(nn.Module):
    def __init__(self, n_input, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_input, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)
        
        # Initialize weights simply
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.dq1 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.dq2 = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d))

class DQNAgent:
    def __init__(self, n_input, n_actions, lr=0.001, gamma=0.99, epsilon=1.0, device='cpu'):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.batch_size = 32
        self.device = device
        
        self.model = DQNNetwork(n_input, n_actions).to(device)
        self.target = DQNNetwork(n_input, n_actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.weights = self._init_weights([0.8, 0.6, 0.6, 0.7, 0.7])

    def _init_weights(self, vals):
        total = sum(vals)
        return np.array([v / total for v in vals])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(st)
        return torch.argmax(q).item()

    def process_rewards(self, state, action, next_state, metrics):
        # Stage 1 analysis
        w = self.weights
        r1 = w[0]*metrics['uptime'] + w[1]*metrics['mem'] + w[2]*metrics['disk']
        
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        nst = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            curr_q = self.model(st)[0, action].item()
            nxt_q = self.target(nst).max().item()
        
        dq1 = 0.001 * (r1 + self.gamma * nxt_q - curr_q)
        self.model.dq1.copy_(torch.tensor(dq1))
        
        # Stage 2 analysis
        r2 = w[3]*metrics['cpu'] + w[4]*metrics['ram']
        dq2 = 0.001 * (r2 + self.gamma * nxt_q - curr_q)
        self.model.dq2.copy_(torch.tensor(dq2))
        
        return dq1, r1, dq2, r2

    def train_step(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0.0
        
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        
        curr_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nxt_q = self.target(ns).max(1)[0]
            target_q = r + (1 - d) * self.gamma * nxt_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
