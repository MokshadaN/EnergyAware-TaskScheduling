import torch
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
from agents.dqn import SchedulingModel

class ExperienceStore:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        
    def push(self, state, action, reward, next_state, done, error=None):
        priority = 1.0 if error is None else (abs(error) + 1e-5) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices)
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

class SystemScheduler:
    def __init__(self, n_input: int, n_actions: int, device: torch.device,
                 learning_rate: float = 0.001, gamma: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_min: float = 0.01,
                 exploration_decay: float = 0.995, batch_size: int = 64):
        self.n_actions = n_actions
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.device = device
        self.model = SchedulingModel(n_input, n_actions, learning_rate, str(device)).to(device)
        self.target_model = SchedulingModel(n_input, n_actions, learning_rate, str(device)).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = ExperienceStore(capacity=20000)
        self.weights = self._calculate_importance_weights([0.8, 0.7, 0.6, 0.7, 0.7, 0.6])
        
    def _calculate_importance_weights(self, importance_values: List[float]) -> np.ndarray:
        total = sum(importance_values)
        weights = np.array([i / total for i in importance_values])
        return weights * 0.95 + 0.05 / len(weights)
    
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.exploration_rate:
            return random.randint(0, self.n_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                values = self.model(state_tensor)
            if self.exploration_rate > 0.1:
                temperature = max(0.5, self.exploration_rate)
                probs = torch.softmax(values / temperature, dim=1)
                return torch.multinomial(probs, 1).item()
            else:
                return torch.argmax(values).item()
    
    def process_utilization_stage(self, state: np.ndarray, action: int, next_state: np.ndarray,
                                 uptime: float, memory_util: float, disk_util: float) -> Tuple[float, float]:
        wU, wM, wD = self.weights[0], self.weights[1], self.weights[2]
        penalty = -0.1 * max(0, memory_util + disk_util - 150)
        R1 = wU * uptime + wM * memory_util + wD * disk_util + penalty
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            current_val = self.model(state_tensor)[0, action].item()
            next_val_max = self.target_model(next_state_tensor).max().item()
        alpha = self.model.learning_rate
        dq1 = alpha * (R1 + self.gamma * next_val_max - current_val)
        self.model.update_correction_factors(dq1, self.model.dq2.item())
        return dq1, R1
    
    def process_load_stage(self, state: np.ndarray, action: int, next_state: np.ndarray,
                          cpu_util: float, ram_util: float) -> Tuple[float, float]:
        wC, wR = self.weights[3], self.weights[4]
        balance = -0.2 * abs(cpu_util - ram_util)
        R2 = wC * cpu_util + wR * ram_util + balance
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            current_val = self.model(state_tensor)[0, action].item()
            next_val_max = self.target_model(next_state_tensor).max().item()
        alpha = self.model.learning_rate
        dq2 = alpha * (R2 + self.gamma * next_val_max - current_val)
        self.model.update_correction_factors(self.model.dq1.item(), dq2)
        return dq2, R2
    
    def update_model(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        experience = self.memory.sample(self.batch_size)
        if experience is None:
            return 0.0
        states, actions, rewards, next_states, dones, indices = experience
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        current_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_values = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_values = rewards + (1 - dones) * self.gamma * next_values
        td_errors = (target_values - current_values.squeeze()).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        loss = self.model.loss_fn(current_values.squeeze(), target_values)
        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.model.optimizer.step()
        self.model.scheduler.step()
        self.soft_update_target_model()
        return loss.item()
    
    def soft_update_target_model(self, tau: float = 0.01):
        for target_param, local_param in zip(self.target_model.parameters(),
                                             self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    
    def store_transition(self, state, action, reward, next_state, done, error=None):
        self.memory.push(state, action, reward, next_state, done, error)
