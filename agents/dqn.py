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
        self.fc1 = nn.Linear(n_input, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.2)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Handle batch size of 1 for BatchNorm compatibility
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        if x.size(0) > 1 or not self.training:
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn3(self.fc3(x)))
        else:
            # Fallback for single sample training (not ideal but avoids crash)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class PrioritizedReplay:
    def __init__(self, capacity=20000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done, error=None):
        p = 1.0 if error is None else (abs(error) + 1e-5) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(p)

    def sample(self, batch_size):
        if not self.buffer: return None
        ps = np.array(self.priorities)
        probs = ps / ps.sum()
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d), indices

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-5) ** self.alpha

class DQNAgent:
    def __init__(self, n_input, n_actions, lr=0.0005, device='cpu'):
        self.n_actions = n_actions
        self.device = device
        self.gamma = 0.99
        self.epsilon = 1.0
        self.memory = PrioritizedReplay()
        
        self.model = DQNNetwork(n_input, n_actions).to(device)
        self.target = DQNNetwork(n_input, n_actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.loss_fn = nn.SmoothL1Loss()
        self.weights = self._init_weights([0.8, 0.7, 0.6, 0.7, 0.7, 0.6])

    def _init_weights(self, vals):
        w = np.array(vals) / sum(vals)
        return w * 0.95 + 0.05 / len(w)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        self.model.eval()
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.model(st)
            if not eval_mode and self.epsilon > 0.1:
                probs = torch.softmax(q / max(0.5, self.epsilon), dim=1)
                return torch.multinomial(probs, 1).item()
            return torch.argmax(q).item()

    def learn(self, batch_size=64):
        if len(self.memory.buffer) < batch_size: return 0.0
        
        self.model.train()
        batch = self.memory.sample(batch_size)
        s, a, r, ns, d, idxs = batch
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        
        curr_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_actions = self.model(ns).argmax(1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions).squeeze()
            target_q = r + (1 - d) * self.gamma * next_q
            
        td_errors = (target_q - curr_q).cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)
        
        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Soft update target
        for tp, lp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(0.01 * lp.data + 0.99 * tp.data)
            
        if self.epsilon > 0.01: self.epsilon *= 0.995
        return loss.item()
