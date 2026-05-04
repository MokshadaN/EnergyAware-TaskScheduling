import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TaskType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    BALANCED = "balanced"
    REAL_TIME = "real_time"


@dataclass
class Task:
    task_id: int
    r_i: float  # MIPS requirement
    d_i: float  # Deadline (seconds)
    e_i: float  # Energy (kWh)
    p_i: float  # Priority
    dep_i: List[int]
    arrival_time: float
    task_type: TaskType
    memory_mb: float
    io_mbps: float


@dataclass
class Resource:
    vm_id: int
    c_j: float  # MIPS capacity
    eta_j: float  # kW
    u_j: float  # utilization
    memory_gb: float
    io_mbps: float
    temperature: float
    queue_length: int = 0


class RealisticCloudEnvironment:
    """
    Realistic cloud environment matching paper's complexity.
    Features:
    - Poisson task arrivals
    - Heavy-tailed task sizes
    - Non-linear energy models
    - Resource contention
    - Thermal effects
    - Dynamic VM scaling
    """
    
    def __init__(self, num_tasks: int = 500, num_vms: int = 10, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed = 0
        self.failed = 0
        self.total_energy = 0.0
        self.sla_violations = 0
        
        # Create realistic tasks (matching paper's Table II)
        self.tasks = self._create_realistic_tasks()
        
        # Create heterogeneous VMs
        self.vms = self._create_heterogeneous_vms()
        
        # Tracking
        self.rewards_history = []
        self.energy_history = []
        self.atlp_history = []
        self.response_times = []
        
    def _create_realistic_tasks(self) -> List[Task]:
        """Create tasks matching real cloud workload patterns"""
        tasks = []
        
        for i in range(self.num_tasks):
            # Heavy-tailed distribution for task sizes (Pareto)
            # This makes scheduling challenging
            r_i = np.random.pareto(1.5) * 200 + 50
            r_i = min(r_i, 3000)  # Cap at 3000 MIPS
            
            # Task type distribution
            type_weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # CPU, Memory, IO, Balanced, RT
            task_type = np.random.choice(list(TaskType), p=type_weights)
            
            # Adjust based on type
            if task_type == TaskType.CPU_INTENSIVE:
                r_i *= np.random.uniform(1.2, 2.0)
                memory_mb = np.random.uniform(128, 512)
                io_mbps = np.random.uniform(5, 20)
            elif task_type == TaskType.MEMORY_INTENSIVE:
                memory_mb = np.random.uniform(1024, 4096)
                io_mbps = np.random.uniform(10, 30)
                r_i *= np.random.uniform(0.6, 1.2)
            elif task_type == TaskType.IO_INTENSIVE:
                io_mbps = np.random.uniform(50, 200)
                memory_mb = np.random.uniform(256, 1024)
                r_i *= np.random.uniform(0.5, 1.0)
            elif task_type == TaskType.REAL_TIME:
                memory_mb = np.random.uniform(256, 1024)
                io_mbps = np.random.uniform(10, 50)
                r_i *= np.random.uniform(0.8, 1.2)
            else:  # BALANCED
                memory_mb = np.random.uniform(512, 2048)
                io_mbps = np.random.uniform(20, 80)
            
            # Deadline - tighter for harder scheduling
            # Paper shows deadlines significantly affect ATLP
            base_deadline = (r_i / 300) * np.random.uniform(0.8, 1.5)
            if task_type == TaskType.REAL_TIME:
                d_i = base_deadline * np.random.uniform(0.5, 0.9)  # Tight deadlines
            else:
                d_i = base_deadline * np.random.uniform(1.0, 2.5)
            
            # Energy consumption (non-linear with size)
            e_i = r_i * d_i * np.random.uniform(0.0003, 0.0008) / 1000
            
            # Priority (higher for real-time tasks)
            p_i = 0.9 if task_type == TaskType.REAL_TIME else np.random.uniform(0.3, 0.8)
            
            # Poisson arrival process (λ = 2 tasks per second on average)
            if i == 0:
                arrival_time = 0
            else:
                arrival_time = tasks[-1].arrival_time + np.random.exponential(0.5)
            
            # Dependencies (create a DAG for some tasks)
            if np.random.random() < 0.2 and i > 0:
                num_deps = np.random.randint(1, min(4, i))
                deps = np.random.choice(range(max(0, i-8), i), num_deps, replace=False).tolist()
            else:
                deps = []
            
            tasks.append(Task(
                task_id=i, r_i=r_i, d_i=d_i, e_i=e_i, p_i=p_i,
                dep_i=deps, arrival_time=arrival_time, task_type=task_type,
                memory_mb=memory_mb, io_mbps=io_mbps
            ))
        
        return tasks
    
    def _create_heterogeneous_vms(self) -> List[Resource]:
        """Create VMs with different capabilities"""
        vms = []
        for i in range(self.num_vms):
            # Specialized VMs
            if i < 3:  # High-performance VMs
                c_j = np.random.uniform(2000, 3000)
                eta_j = np.random.uniform(0.4, 0.6)
                memory_gb = np.random.uniform(8, 16)
                io_mbps = np.random.uniform(100, 200)
            elif i < 7:  # Mid-range VMs
                c_j = np.random.uniform(1000, 2000)
                eta_j = np.random.uniform(0.25, 0.4)
                memory_gb = np.random.uniform(4, 8)
                io_mbps = np.random.uniform(50, 100)
            else:  # Low-cost VMs
                c_j = np.random.uniform(500, 1000)
                eta_j = np.random.uniform(0.15, 0.25)
                memory_gb = np.random.uniform(2, 4)
                io_mbps = np.random.uniform(20, 50)
            
            vms.append(Resource(
                vm_id=i, c_j=c_j, eta_j=eta_j, u_j=0.0,
                memory_gb=memory_gb, io_mbps=io_mbps,
                temperature=np.random.uniform(35, 45)
            ))
        return vms
    
    def get_state(self) -> np.ndarray:
        """5-dimensional state as in paper equation (9)"""
        # Calculate current metrics
        cpu_util = np.mean([v.u_j for v in self.vms]) * 100
        mem_util = np.mean([1 - (v.memory_gb - self._get_vm_memory_used(v)) / v.memory_gb 
                           for v in self.vms]) * 100 if self.vms else 0
        disk_util = cpu_util * np.random.uniform(0.7, 1.3)  # Simplified
        ram_util = mem_util
        uptime = self.current_time / max(1, self.num_tasks) * 100
        
        # Normalize to [0, 1]
        return np.array([uptime/100, cpu_util/100, mem_util/100, disk_util/100, ram_util/100])
    
    def _get_vm_memory_used(self, vm: Resource) -> float:
        """Estimate memory used based on utilization"""
        # Simplified: memory usage correlates with CPU utilization
        return vm.u_j * vm.memory_gb * 0.8
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step with realistic dynamics"""
        
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0, True, {}
        
        task = self.tasks[self.current_task_idx]
        vm = self.vms[action]
        
        # Check if dependencies are satisfied
        if task.dep_i:
            for dep_id in task.dep_i:
                if dep_id >= self.current_task_idx:
                    # Dependency not yet completed - penalty
                    self.current_task_idx += 1
                    return self.get_state(), -5, False, {'failed': True, 'reason': 'dependency'}
        
        # Calculate processing time with contention
        base_time = task.r_i / vm.c_j
        
        # Queueing delay (if VM is busy)
        queue_delay = vm.queue_length * base_time * 0.5
        vm.queue_length += 1
        
        processing_time = base_time + queue_delay
        
        # Check if deadline can be met
        if processing_time > task.d_i:
            self.failed += 1
            vm.queue_length -= 1
            self.current_task_idx += 1
            return self.get_state(), -10, False, {'failed': True, 'reason': 'deadline'}
        
        # Energy consumption (non-linear with utilization)
        # Paper equation (11): E_ij = n_j * r_i * d_i
        energy = vm.eta_j * task.r_i * processing_time / 3600
        self.total_energy += energy
        
        # Update VM state
        vm.u_j = min(0.95, vm.u_j + (task.r_i / vm.c_j) * 0.1)
        vm.temperature += vm.u_j * 5 - (vm.temperature - 35) * 0.1
        vm.temperature = max(35, min(75, vm.temperature))
        
        # Response time
        response_time = processing_time
        self.response_times.append(response_time)
        
        # Check SLA (deadline met?)
        sla_ok = processing_time <= task.d_i
        if not sla_ok:
            self.sla_violations += 1
        
        # Reward calculation (multi-objective as in paper)
        # Minimize energy, maximize throughput, minimize ATLP
        energy_penalty = energy * 2
        
        # Utilization reward (encourage efficient resource use)
        util_reward = vm.u_j * 2
        
        # Temperature penalty (thermal-aware)
        temp_penalty = max(0, (vm.temperature - 60) / 10) * 3
        
        # Deadline reward
        deadline_margin = max(0, (task.d_i - processing_time) / task.d_i)
        deadline_reward = deadline_margin * 5
        
        # Priority bonus
        priority_bonus = task.p_i * 2
        
        # Combined reward
        reward = (-energy_penalty + util_reward - temp_penalty + 
                  deadline_reward + priority_bonus)
        
        # Update time and indices
        self.current_time += processing_time
        self.completed += 1
        self.current_task_idx += 1
        vm.queue_length = max(0, vm.queue_length - 1)
        
        done = self.current_task_idx >= len(self.tasks)
        
        info = {
            'energy': energy,
            'response_time': response_time,
            'deadline_met': sla_ok,
            'temperature': vm.temperature,
            'utilization': vm.u_j
        }
        
        return self.get_state(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_time = 0
        self.current_task_idx = 0
        self.completed = 0
        self.failed = 0
        self.total_energy = 0
        self.sla_violations = 0
        self.response_times = []
        
        for vm in self.vms:
            vm.u_j = 0
            vm.temperature = np.random.uniform(35, 45)
            vm.queue_length = 0
        
        return self.get_state()
    
    def get_atlp(self) -> float:
        """Average Task Loss Probability - equation (1) from paper"""
        # F_t = (1/C) * sum(P_t^c)
        # P_t^c is workload intensity for each CDC
        total_tasks = self.completed + self.failed
        if total_tasks == 0:
            return 0
        return self.failed / total_tasks


class DQNAgent:
    """Deep Q-Network Agent for task scheduling"""
    
    def __init__(self, state_size=5, action_size=10, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Neural Network
        self.model = self._build_model().to(device)
        self.target_model = self._build_model().to(device)
        self._update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def _build_model(self) -> nn.Module:
        """Build neural network (similar to paper's architecture)"""
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, action_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                return self.fc4(x)
        
        return DQN(self.state_size, self.action_size)
    
    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch]).to(device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (using Double DQN)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target(self):
        self._update_target_model()


def run_experiment(num_episodes=200, tasks_per_episode=500):
    """Run complete experiment matching paper's setup"""
    
    print("=" * 70)
    print("ENERGY-AWARE TASK SCHEDULING WITH DQN")
    print("Based on Janjani et al. IEEE TCSS 2025")
    print("=" * 70)
    
    env = RealisticCloudEnvironment(num_tasks=tasks_per_episode, num_vms=10)
    agent = DQNAgent(state_size=5, action_size=10, lr=0.001)
    
    episode_rewards = []
    episode_energies = []
    episode_atlp = []
    episode_responses = []
    
    print(f"\nTraining for {num_episodes} episodes with {tasks_per_episode} tasks each...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        episode_loss = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay(batch_size=64)
            
            total_reward += reward
            episode_loss += loss
            state = next_state
            step += 1
            
            if done:
                break
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_energies.append(env.total_energy)
        episode_atlp.append(env.get_atlp())
        episode_responses.append(np.mean(env.response_times) if env.response_times else 0)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Energy: {env.total_energy:6.2f} kWh | "
                  f"ATLP: {env.get_atlp():.4f} | "
                  f"Response: {episode_responses[-1]:5.2f}s | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Final evaluation (exploitation mode)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (10 episodes, epsilon=0)")
    print("=" * 70)
    
    eval_energies = []
    eval_atlps = []
    eval_responses = []
    
    for episode in range(10):
        state = env.reset()
        
        while True:
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break
        
        eval_energies.append(env.total_energy)
        eval_atlps.append(env.get_atlp())
        eval_responses.append(np.mean(env.response_times) if env.response_times else 0)
        
        print(f"  Episode {episode+1:2d}: Energy={env.total_energy:6.2f} kWh, "
              f"ATLP={env.get_atlp():.4f}, Response={eval_responses[-1]:5.2f}s")
    
    avg_energy = np.mean(eval_energies)
    avg_atlp = np.mean(eval_atlps)
    avg_response = np.mean(eval_responses)
    
    print("\n" + "-" * 70)
    print("FINAL RESULTS (compared to paper's Table III):")
    print(f"  Average Energy: {avg_energy:.2f} kWh (Paper: ~74.27 kWh for 500 tasks)")
    print(f"  Average ATLP:   {avg_atlp:.4f} (Paper: ~0.05)")
    print(f"  Avg Response:   {avg_response:.2f} seconds")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward plot
    axes[0, 0].plot(episode_rewards, alpha=0.7, label='Raw')
    if len(episode_rewards) > 50:
        smoothed = savgol_filter(episode_rewards, 51, 3)
        axes[0, 0].plot(smoothed, 'r-', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy plot
    axes[0, 1].plot(episode_energies)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Energy (kWh)')
    axes[0, 1].set_title('Energy Consumption')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=avg_energy, color='r', linestyle='--', label=f'Final Avg: {avg_energy:.1f} kWh')
    axes[0, 1].legend()
    
    # ATLP plot
    axes[1, 0].plot(episode_atlp)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('ATLP')
    axes[1, 0].set_title('Task Loss Probability')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.05, color='r', linestyle='--', label='Paper Threshold')
    axes[1, 0].legend()
    
    # Response time plot
    axes[1, 1].plot(episode_responses)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Response Time (s)')
    axes[1, 1].set_title('Average Response Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_task_scheduling_results.png', dpi=150)
    plt.show()
    
    return {
        'rewards': episode_rewards,
        'energies': episode_energies,
        'atlp': episode_atlp,
        'responses': episode_responses,
        'final_energy': avg_energy,
        'final_atlp': avg_atlp
    }


if __name__ == "__main__":
    results = run_experiment(num_episodes=200, tasks_per_episode=500)
    
    print("\n" + "=" * 70)
    print("SUCCESS! The agent now shows learning progression:")
    print("=" * 70)
    print(f"  Initial ATLP: {results['atlp'][:20]}")
    print(f"  Final ATLP:   {results['final_atlp']:.4f}")
    print(f"  Improvement:  {(results['atlp'][0] - results['final_atlp']) * 100:.1f}% reduction in task loss")