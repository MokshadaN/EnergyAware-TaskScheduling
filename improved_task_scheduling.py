# improved_task_scheduling.py
"""
Enhanced Energy-Aware Task Scheduling using Deep Q-Learning
with comprehensive metrics and visualizations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set seeds
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class Task:
    """Task characteristics with realistic distributions"""
    task_id: int
    r_i: float  # Resource requirements in MIPS
    d_i: float  # Deadline in seconds
    e_i: float  # Energy consumption in kWh
    p_i: float  # Priority
    dep_i: List[int] = field(default_factory=list)
    arrival_time: float = 0.0  # For realistic workload simulation


@dataclass
class Resource:
    """VM resources with heterogeneous characteristics"""
    vm_id: int
    c_j: float  # Capacity in MIPS
    eta_j: float  # Energy rate in kW
    u_j: float = 0.0  # Current utilization
    power_state: str = "active"  # active, idle, sleep


class EnhancedTaskSchedulingDQN(nn.Module):
    """Enhanced DQN with dropout and batch normalization for stability"""
    
    def __init__(self, n_input: int, n_actions: int, learning_rate: float = 0.001):
        super().__init__()
        
        self.n_input = n_input
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Enhanced network architecture with batch norm and dropout
        self.network = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self._initialize_weights()
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for robustness
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better sample efficiency"""
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, error=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class EnhancedAgent:
    """Enhanced agent with improved learning mechanisms"""
    
    def __init__(self, n_input: int, n_actions: int, learning_rate: float = 0.0003,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995, batch_size: int = 64,
                 target_update_freq: int = 200, soft_update_tau: float = 0.005):
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau
        self.step_count = 0
        
        # Networks
        self.q_network = EnhancedTaskSchedulingDQN(n_input, n_actions, learning_rate).to(device)
        self.target_network = EnhancedTaskSchedulingDQN(n_input, n_actions, learning_rate).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        
        # Metric weights (optimized based on empirical results)
        self.metric_weights = {
            'uptime': 0.25,
            'cpu': 0.25,
            'memory': 0.15,
            'disk': 0.15,
            'ram': 0.20
        }
        
        # Tracking for analysis
        self.q_values_history = []
        self.loss_history = []
        self.epsilon_history = []
        
    def choose_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy with optional evaluation mode"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        self.q_network.eval()  # Set to eval mode for inference
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        if not eval_mode:
            self.q_values_history.append(q_values.cpu().numpy())
            self.q_network.train()  # Switch back to train mode
        
        return torch.argmax(q_values).item()
    
    def compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray,
                       energy_delta: float, task_priority: float) -> float:
        """
        Compute multi-objective reward combining:
        1. Energy efficiency
        2. Resource utilization balance
        3. Task priority satisfaction
        """
        # Energy reward (negative because we want to minimize)
        energy_reward = -energy_delta * 10
        
        # Utilization balance reward (penalize imbalance)
        cpu_util, mem_util, disk_util = next_state[1], next_state[2], next_state[3]
        util_variance = np.var([cpu_util, mem_util, disk_util])
        balance_reward = -util_variance * 0.1
        
        # Priority reward
        priority_reward = task_priority * 0.5
        
        # Completion bonus
        completion_bonus = 1.0 if energy_delta > 0 else 0.0
        
        total_reward = (energy_reward * 0.6 + balance_reward * 0.2 + 
                       priority_reward * 0.1 + completion_bonus * 0.1)
        
        return total_reward
    
    def learn(self) -> Tuple[float, float]:
        """Enhanced learning with prioritized replay and Double DQN"""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        self.q_network.train()  # Ensure training mode
        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Double DQN: use q_network to select action, target_network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # TD errors for priority update
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights * (target_q_values - current_q_values) ** 2).mean()
        
        # Backpropagation
        self.q_network.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_network.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(self.soft_update_tau * local_param.data + 
                                   (1 - self.soft_update_tau) * target_param.data)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self.loss_history.append(loss.item())
        
        return loss.item(), np.mean(np.abs(td_errors))
    
    def save_model(self, path: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.epsilon = checkpoint['epsilon']


class RealisticCloudEnvironment:
    """Enhanced cloud environment with realistic workload patterns"""
    
    def __init__(self, num_tasks: int = 500, num_vms: int = 10, 
                 workload_pattern: str = "mixed"):
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        self.workload_pattern = workload_pattern
        
        self.tasks = self._generate_realistic_tasks()
        self.resources = self._generate_heterogeneous_resources()
        
        # Tracking
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.failed_tasks = []  # Tasks that missed deadline
        self.total_energy = 0.0
        self.idle_energy = 0.0
        
        # Historical data
        self.resource_util_history = []
        self.energy_history = []
        self.sla_violations = 0
        
    def _generate_realistic_tasks(self):
        """Generate tasks with realistic distributions (bimodal for real-world)"""
        tasks = []
        arrival_time = 0
        
        for i in range(self.num_tasks):
            # Bimodal task sizes (small vs large)
            if random.random() < 0.7:
                r_i = random.gauss(200, 50)  # Small tasks (70%)
            else:
                r_i = random.gauss(800, 150)  # Large tasks (30%)
            r_i = max(50, min(1500, r_i))
            
            # Deadlines correlated with size
            d_i = r_i / 10 + random.gauss(0, 20)
            d_i = max(10, d_i)
            
            # Arrival times (Poisson process)
            arrival_time += np.random.exponential(2)
            
            # Energy consumption proportional to size
            e_i = r_i * 0.001 * random.uniform(0.8, 1.2)
            
            tasks.append(Task(
                task_id=i,
                r_i=r_i,
                d_i=d_i,
                e_i=e_i,
                p_i=random.uniform(0, 1),
                arrival_time=arrival_time
            ))
        
        return tasks
    
    def _generate_heterogeneous_resources(self):
        """Generate heterogeneous VMs"""
        resources = []
        for i in range(self.num_vms):
            # Different VM types
            if i < 3:
                # High-performance VMs
                c_j = random.gauss(2000, 200)
                eta_j = random.gauss(0.5, 0.1)
            elif i < 7:
                # Mid-range VMs
                c_j = random.gauss(1000, 150)
                eta_j = random.gauss(0.3, 0.05)
            else:
                # Low-power VMs
                c_j = random.gauss(500, 100)
                eta_j = random.gauss(0.15, 0.03)
            
            resources.append(Resource(vm_id=i, c_j=max(100, c_j), eta_j=max(0.05, eta_j)))
        
        return resources
    
    def get_state(self) -> np.ndarray:
        """Get normalized state vector"""
        # Calculate metrics
        cpu_util = self._get_cpu_utilization() / 100.0
        mem_util = self._get_memory_utilization() / 100.0
        disk_util = self._get_disk_utilization() / 100.0
        
        # Normalized time progression
        time_progress = self.current_task_idx / max(1, self.num_tasks)
        
        # Queue length normalized
        queue_length = min(1.0, self.current_task_idx / max(1, self.num_tasks))
        
        return np.array([time_progress, cpu_util, mem_util, disk_util, queue_length])
    
    def _get_cpu_utilization(self) -> float:
        if not self.resources:
            return 0.0
        total_load = sum(r.u_j for r in self.resources)
        return (total_load / len(self.resources)) * 100
    
    def _get_memory_utilization(self) -> float:
        base_util = 30 + 60 * (self.current_task_idx / max(1, self.num_tasks))
        return min(95, base_util + random.gauss(0, 5))
    
    def _get_disk_utilization(self) -> float:
        base_util = 20 + 50 * (self.current_task_idx / max(1, self.num_tasks))
        return min(80, base_util + random.gauss(0, 3))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step with realistic dynamics"""
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0.0, True, {}
        
        current_task = self.tasks[self.current_task_idx]
        selected_vm = self.resources[action % len(self.resources)]
        
        # Check if VM can handle the task
        if current_task.r_i > selected_vm.c_j:
            # Task migrates to another VM
            suitable_vms = [vm for vm in self.resources if vm.c_j >= current_task.r_i]
            if suitable_vms:
                selected_vm = min(suitable_vms, key=lambda v: v.u_j)
            else:
                self.failed_tasks.append(current_task)
                self.current_task_idx += 1
                return self.get_state(), -5.0, False, {'failed': True}
        
        # Calculate execution time
        exec_time = current_task.r_i / selected_vm.c_j
        
        # Check deadline
        if self.current_time + exec_time > current_task.d_i:
            self.sla_violations += 1
            deadline_penalty = -2.0
        else:
            deadline_penalty = 0.0
        
        # Calculate energy (dynamic power + idle power)
        active_power = selected_vm.eta_j * (0.7 + 0.3 * (current_task.r_i / selected_vm.c_j))
        idle_power = selected_vm.eta_j * 0.3
        task_energy = active_power * (exec_time / 3600)
        self.idle_energy += idle_power * (exec_time / 3600)
        self.total_energy += task_energy
        
        # Update VM utilization (with decay for previous tasks)
        for vm in self.resources:
            vm.u_j = max(0, vm.u_j - 0.05)  # Gradual decay
        selected_vm.u_j = min(1.0, selected_vm.u_j + (current_task.r_i / selected_vm.c_j) * 0.3)
        
        # Record history
        self.resource_util_history.append([vm.u_j for vm in self.resources])
        self.energy_history.append(self.total_energy)
        
        # Compute reward
        reward = self._compute_reward(task_energy, deadline_penalty, current_task.p_i)
        
        # Update state
        self.current_task_idx += 1
        self.current_time += exec_time
        self.completed_tasks.append(current_task)
        
        done = self.current_task_idx >= len(self.tasks)
        next_state = self.get_state()
        
        info = {
            'task_id': current_task.task_id,
            'vm_id': selected_vm.vm_id,
            'energy': task_energy,
            'exec_time': exec_time,
            'deadline_met': deadline_penalty == 0,
            'utilization': selected_vm.u_j
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, energy: float, deadline_penalty: float, priority: float) -> float:
        """Compute reward with multiple objectives"""
        energy_score = -energy * 100  # Minimize energy
        deadline_score = deadline_penalty
        progress_score = (self.current_task_idx / max(1, self.num_tasks)) * 0.1
        priority_score = priority * 0.2
        
        return energy_score + deadline_score + progress_score + priority_score
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.failed_tasks = []
        self.total_energy = 0.0
        self.idle_energy = 0.0
        self.sla_violations = 0
        self.resource_util_history = []
        self.energy_history = []
        
        for resource in self.resources:
            resource.u_j = 0.0
        
        return self.get_state()
    
    def get_comprehensive_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        tasks_processed = len(self.completed_tasks)
        
        return {
            'total_energy_kwh': self.total_energy,
            'idle_energy_kwh': self.idle_energy,
            'active_energy_kwh': self.total_energy - self.idle_energy,
            'energy_per_task': self.total_energy / max(1, tasks_processed),
            'sla_violations': self.sla_violations,
            'sla_violation_rate': self.sla_violations / max(1, self.num_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': (self.num_tasks - len(self.failed_tasks)) / self.num_tasks,
            'throughput': tasks_processed / max(1, self.current_time),
            'avg_cpu_util': np.mean([np.mean(h) for h in self.resource_util_history]) * 100 if self.resource_util_history else 0,
            'resource_efficiency': (self.total_energy - self.idle_energy) / max(1, self.total_energy),
            'makespan': self.current_time
        }


def train_with_metrics(env: RealisticCloudEnvironment, agent: EnhancedAgent, 
                       num_episodes: int = 200, eval_freq: int = 20) -> Dict:
    """Enhanced training with comprehensive metrics collection"""
    
    training_stats = {
        'episode_rewards': [],
        'episode_energies': [],
        'episode_losses': [],
        'episode_td_errors': [],
        'episode_sla_rate': [],
        'episode_throughput': [],
        'eval_metrics': []
    }
    
    best_energy = float('inf')
    best_model_path = 'best_model.pth'
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        episode_td_errors = []
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn
            loss, td_error = agent.learn()
            episode_losses.append(loss)
            episode_td_errors.append(td_error)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Record metrics
        metrics = env.get_comprehensive_metrics()
        training_stats['episode_rewards'].append(total_reward)
        training_stats['episode_energies'].append(metrics['total_energy_kwh'])
        training_stats['episode_losses'].append(np.mean(episode_losses))
        training_stats['episode_td_errors'].append(np.mean(episode_td_errors))
        training_stats['episode_sla_rate'].append(metrics['sla_violation_rate'])
        training_stats['episode_throughput'].append(metrics['throughput'])
        
        # Save best model
        if metrics['total_energy_kwh'] < best_energy and episode > num_episodes // 2:
            best_energy = metrics['total_energy_kwh']
            agent.save_model(best_model_path)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent_episode(env, agent, num_episodes=5)
            training_stats['eval_metrics'].append({
                'episode': episode,
                'energy': eval_metrics['avg_energy'],
                'sla_rate': eval_metrics['avg_sla_rate']
            })
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Energy: {metrics['total_energy_kwh']:.1f} kWh | "
                  f"SLA: {metrics['sla_violation_rate']:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {np.mean(episode_losses):.4f}")
    
    return training_stats


def evaluate_agent_episode(env: RealisticCloudEnvironment, agent: EnhancedAgent, 
                           num_episodes: int = 10) -> Dict:
    """Evaluate agent over multiple episodes"""
    all_metrics = []
    
    for _ in range(num_episodes):
        state = env.reset()
        
        while True:
            action = agent.choose_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if done:
                break
        
        all_metrics.append(env.get_comprehensive_metrics())
    
    # Aggregate results
    return {
        'avg_energy': np.mean([m['total_energy_kwh'] for m in all_metrics]),
        'std_energy': np.std([m['total_energy_kwh'] for m in all_metrics]),
        'avg_sla_rate': np.mean([m['sla_violation_rate'] for m in all_metrics]),
        'avg_throughput': np.mean([m['throughput'] for m in all_metrics]),
        'avg_success_rate': np.mean([m['success_rate'] for m in all_metrics]),
        'avg_resource_efficiency': np.mean([m['resource_efficiency'] for m in all_metrics])
    }


def run_baseline_algorithms(env: RealisticCloudEnvironment, num_tasks: int) -> Dict:
    """Run baseline algorithms for comparison"""
    
    def run_fcfs():
        env.reset()
        for i in range(env.num_tasks):
            # Round-robin among VMs
            env.step(i % env.num_vms)
        return env.get_comprehensive_metrics()
    
    def run_edf():
        env.reset()
        # Sort tasks by deadline
        original_tasks = env.tasks
        env.tasks = sorted(env.tasks, key=lambda t: t.d_i)
        for i in range(env.num_tasks):
            env.step(i % env.num_vms)
        env.tasks = original_tasks
        return env.get_comprehensive_metrics()
    
    def run_min_min():
        env.reset()
        for i in range(env.num_tasks):
            # Assign to VM with minimum completion time
            vm_completion_times = []
            for j in range(env.num_vms):
                exec_time = env.tasks[i].r_i / env.resources[j].c_j
                vm_completion_times.append(exec_time)
            best_vm = np.argmin(vm_completion_times)
            env.step(best_vm)
        return env.get_comprehensive_metrics()
    
    def run_max_min():
        env.reset()
        for i in range(env.num_tasks):
            # Assign to VM with maximum completion time (load balancing)
            vm_completion_times = []
            for j in range(env.num_vms):
                exec_time = env.tasks[i].r_i / env.resources[j].c_j
                vm_completion_times.append(exec_time)
            best_vm = np.argmax(vm_completion_times)
            env.step(best_vm)
        return env.get_comprehensive_metrics()
    
    print(f"  Running baselines for {num_tasks} tasks...")
    
    return {
        'FCFS': run_fcfs(),
        'EDF': run_edf(),
        'Min-Min': run_min_min(),
        'Max-Min': run_max_min()
    }

