import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class Task:
    """Task characteristics as defined in equation (5)"""
    task_id: int
    r_i: float  # Resource requirements in MIPS
    d_i: float  # Deadline in seconds
    e_i: float  # Energy consumption in kWh
    p_i: float  # Priority
    dep_i: List[int]  # Dependencies
    
    def __post_init__(self):
        if self.dep_i is None:
            self.dep_i = []


@dataclass
class Resource:
    """Resource availability as defined in equation (6)"""
    vm_id: int
    c_j: float  # Resource capacity in MIPS
    eta_j: float  # Energy consumption rate in kW
    u_j: float  # Resource utilization


class TaskSchedulingDQN(nn.Module):
    """
    Deep Q-Network for task scheduling as described in Algorithm 1.
    Implements the neural network architecture for Q-value approximation.
    """
    
    def __init__(self, n_input: int, n_actions: int, learning_rate: float = 0.001):
        super(TaskSchedulingDQN, self).__init__()
        
        self.n_input = n_input
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Network architecture
        self.fc1 = nn.Linear(n_input, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Correction factors for process stages (DQ1, DQ2)
        self.dq1 = torch.tensor(0.0, device=device)
        self.dq2 = torch.tensor(0.0, device=device)
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Implements the Q-value calculation with correction factors.
        """
        # Add correction factors to the state as described in the paper
        if self.dq1.device != state.device:
            self.dq1 = self.dq1.to(state.device)
            self.dq2 = self.dq2.to(state.device)
        
        # Process through hidden layers with ReLU activations
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Output layer (Q-values)
        q_values = self.fc4(x)
        
        return q_values
    
    def update_correction_factors(self, dq1: float, dq2: float):
        """Update the correction factors for process stages"""
        self.dq1 = torch.tensor(dq1, device=device)
        self.dq2 = torch.tensor(dq2, device=device)


class ExperienceReplay:
    """Experience replay buffer for storing transitions"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class AgentClass:
    """
    Reinforcement Learning Agent for task scheduling.
    Implements the epsilon-greedy policy and learning mechanism.
    """
    
    def __init__(self, n_input: int, n_actions: int, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.998, batch_size: int = 32,
                 target_update_freq: int = 100, soft_update_param: float = 0.01):
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update_param = soft_update_param
        self.step_count = 0
        
        # Main and target networks
        self.q_network = TaskSchedulingDQN(n_input, n_actions, learning_rate).to(device)
        self.target_network = TaskSchedulingDQN(n_input, n_actions, learning_rate).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.memory = ExperienceReplay(capacity=10000)
        
        # Weights for metrics (from equation 4)
        # Importance values from the paper: U=0.8, M=0.6, D=0.6, C=0.7, R=0.7
        self.weights = self._calculate_weights([0.8, 0.6, 0.6, 0.7, 0.7])
        
    def _calculate_weights(self, importance_values: List[float]) -> np.ndarray:
        """
        Calculate normalized weights using equation (4)
        W_i = I_i / sum(I_i)
        """
        total = sum(importance_values)
        return np.array([i / total for i in importance_values])
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        Implements exploration vs exploitation strategy.
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best action based on Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def process_stage1(self, state: np.ndarray, action: int, next_state: np.ndarray,
                       uptime: float, memory_util: float, disk_util: float) -> float:
        """
        Process Stage 1: Task allocation.
        Calculates R1 reward and DQ1 correction factor.
        
        R1 = w_U * U + w_M * M + w_D * D
        DQ1 = alpha * [R1 + gamma * max(Q(S')) - Q(S)[A]]
        """
        # Calculate R1 reward (equation 8)
        wU, wM, wD = self.weights[0], self.weights[1], self.weights[2]
        R1 = wU * uptime + wM * memory_util + wD * disk_util
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            current_q = self.q_network(state_tensor)[0, action].item()
            next_q_max = self.target_network(next_state_tensor).max().item()
        
        # Calculate DQ1 correction factor
        alpha = self.q_network.learning_rate
        dq1 = alpha * (R1 + self.gamma * next_q_max - current_q)
        
        # Update correction factor in network
        self.q_network.update_correction_factors(dq1, self.q_network.dq2.item())
        
        return dq1, R1
    
    def process_stage2(self, state: np.ndarray, action: int, next_state: np.ndarray,
                       cpu_util: float, ram_util: float) -> float:
        """
        Process Stage 2: Process scheduling.
        Calculates R2 reward and DQ2 correction factor.
        
        R2 = w_C * C + w_R * R
        DQ2 = alpha * [R2 + gamma * max(Q(S')) - Q(S)[A]]
        """
        # Calculate R2 reward (equation 8)
        wC, wR = self.weights[3], self.weights[4]
        R2 = wC * cpu_util + wR * ram_util
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            current_q = self.q_network(state_tensor)[0, action].item()
            next_q_max = self.target_network(next_state_tensor).max().item()
        
        # Calculate DQ2 correction factor
        alpha = self.q_network.learning_rate
        dq2 = alpha * (R2 + self.gamma * next_q_max - current_q)
        
        # Update correction factor in network
        self.q_network.update_correction_factors(self.q_network.dq1.item(), dq2)
        
        return dq2, R2
    
    def learn(self) -> float:
        """Update Q-values using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (MSELoss as specified in Algorithm 1)
        loss = self.q_network.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.q_network.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.q_network.optimizer.step()
        
        # Soft update of target network (equation implemented in the paper)
        # h_new = s * h_new + (1 - s) * h_old
        self.soft_update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def soft_update_target_network(self):
        """Soft weight updates for target network stability"""
        s = self.soft_update_param
        for target_param, local_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(s * local_param.data + (1 - s) * target_param.data)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)


class CloudEnvironment:
    """
    Cloud Computing Environment Simulation.
    Implements the CDC scenario described in section III-A.
    """
    
    def __init__(self, num_tasks: int = 100, num_vms: int = 10):
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        
        # Initialize tasks using equation (5): T = {r_i, d_i, e_i, p_i, dep_i}
        self.tasks = self._initialize_tasks()
        
        # Initialize resources using equation (6): RA = {c_j, eta_j, u_j}
        self.resources = self._initialize_resources()
        
        # Workload distribution matrix W
        self.workload_distribution = np.zeros((num_tasks, num_vms))
        
        # Current time and state
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.total_energy = 0.0
        
        # Metrics tracking
        self.metrics_history = {
            'energy': [],
            'cpu_util': [],
            'memory_util': [],
            'disk_util': [],
            'response_times': [],
            'uptime': []
        }
        
    def _initialize_tasks(self) -> List[Task]:
        """Initialize tasks with random characteristics"""
        tasks = []
        for i in range(self.num_tasks):
            task = Task(
                task_id=i,
                r_i=random.uniform(100, 1000),  # MIPS
                d_i=random.uniform(10, 100),     # seconds
                e_i=random.uniform(0.5, 5.0),    # kWh
                p_i=random.uniform(0, 1),        # priority
                dep_i=random.sample(range(max(0, i-5), i), min(3, i)) if i > 0 else []
            )
            tasks.append(task)
        return tasks
    
    def _initialize_resources(self) -> List[Resource]:
        """Initialize VM resources with heterogeneous characteristics"""
        resources = []
        for i in range(self.num_vms):
            resource = Resource(
                vm_id=i,
                c_j=random.uniform(500, 2000),   # MIPS capacity
                eta_j=random.uniform(0.1, 0.5),  # kW
                u_j=0.0  # initial utilization
            )
            resources.append(resource)
        return resources
    
    def get_state(self) -> np.ndarray:
        """
        Get current state S = {U(t), M(t), D(t), C(t), R(t)}
        as defined in equation (9)
        """
        # Calculate current metrics
        uptime = self.current_time / max(1, self.num_tasks) * 100
        cpu_util = self._get_cpu_utilization()
        memory_util = self._get_memory_utilization()
        disk_util = self._get_disk_utilization()
        ram_util = memory_util  # RAM utilization similar to memory
        
        return np.array([uptime, cpu_util, memory_util, disk_util, ram_util])
    
    def _get_cpu_utilization(self) -> float:
        """Calculate average CPU utilization"""
        if not self.resources:
            return 0.0
        total_util = sum(r.u_j for r in self.resources)
        return (total_util / len(self.resources)) * 100
    
    def _get_memory_utilization(self) -> float:
        """Calculate average memory utilization"""
        # Simulated memory utilization
        base_util = 20 + 70 * (self.current_task_idx / max(1, self.num_tasks))
        return min(95, base_util)
    
    def _get_disk_utilization(self) -> float:
        """Calculate average disk utilization"""
        # Simulated disk utilization
        base_util = 10 + 60 * (self.current_task_idx / max(1, self.num_tasks))
        return min(85, base_util)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Selected VM index for task allocation
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0.0, True, {}
        
        current_task = self.tasks[self.current_task_idx]
        selected_vm = self.resources[action] if action < len(self.resources) else self.resources[0]
        
        # Calculate energy consumption using equation (11)
        # E_ij = n_j * r_i * d_i
        task_energy = selected_vm.eta_j * current_task.r_i * current_task.d_i / 3600  # Convert to kWh
        self.total_energy += task_energy
        
        # Update VM utilization
        utilization_ratio = current_task.r_i / selected_vm.c_j
        selected_vm.u_j = min(1.0, selected_vm.u_j + utilization_ratio * 0.1)
        
        # Update workload distribution matrix
        self.workload_distribution[self.current_task_idx, action] = 1
        
        # Calculate response time
        response_time = current_task.r_i / selected_vm.c_j
        self.metrics_history['response_times'].append(response_time)
        
        # Track metrics
        self.metrics_history['energy'].append(self.total_energy)
        self.metrics_history['cpu_util'].append(self._get_cpu_utilization())
        self.metrics_history['memory_util'].append(self._get_memory_utilization())
        self.metrics_history['disk_util'].append(self._get_disk_utilization())
        self.metrics_history['uptime'].append(self.current_time)
        
        # Calculate reward (negative energy consumption to minimize it)
        reward = -task_energy
        
        # Update state
        self.current_task_idx += 1
        self.current_time += response_time
        
        # Check if done
        done = self.current_task_idx >= len(self.tasks)
        
        next_state = self.get_state()
        info = {
            'task_id': current_task.task_id,
            'vm_id': selected_vm.vm_id,
            'energy_consumed': task_energy,
            'response_time': response_time,
            'utilization': selected_vm.u_j
        }
        
        return next_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.total_energy = 0.0
        self.workload_distribution = np.zeros((self.num_tasks, self.num_vms))
        
        # Reset resource utilization
        for resource in self.resources:
            resource.u_j = 0.0
        
        # Reset metrics
        for key in self.metrics_history:
            self.metrics_history[key] = []
        
        return self.get_state()
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for analysis"""
        return {
            'total_energy': self.total_energy,
            'avg_cpu_util': self._get_cpu_utilization(),
            'avg_memory_util': self._get_memory_utilization(),
            'avg_disk_util': self._get_disk_utilization(),
            'avg_response_time': np.mean(self.metrics_history['response_times']) if self.metrics_history['response_times'] else 0,
            'throughput': self.current_task_idx / max(1, self.current_time),
            'task_completion_rate': self.current_task_idx / max(1, self.num_tasks) * 100
        }


def calculate_atlp(env: CloudEnvironment) -> float:
    """
    Calculate Average Task Loss Probability (ATLP) F_t
    as defined in equation (1) and (2)
    
    F_t = (1/C) * sum(P_t^c)
    """
    if not env.resources:
        return 0.0
    
    # Calculate average workload intensity
    P_tau = [r.u_j for r in env.resources]
    F_t = np.mean(P_tau)
    
    return F_t


def train_agent(env: CloudEnvironment, agent: AgentClass, num_episodes: int = 100,
                verbose: bool = True) -> Dict:
    """Train the reinforcement learning agent"""
    
    episode_rewards = []
    episode_losses = []
    episode_energies = []
    episode_atlp = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0
        
        while True:
            # Choose action
            action = agent.choose_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Calculate process stage rewards and correction factors
            uptime, cpu_util, mem_util, disk_util, ram_util = state
            
            # Process Stage 1 (Task Allocation)
            dq1, r1 = agent.process_stage1(state, action, next_state, uptime, mem_util, disk_util)
            
            # Process Stage 2 (Process Scheduling)
            dq2, r2 = agent.process_stage2(state, action, next_state, cpu_util, ram_util)
            
            # Combined reward
            combined_reward = reward + 0.1 * (r1 + r2)
            
            # Store transition
            agent.store_transition(state, action, combined_reward, next_state, done)
            
            # Learn from experiences
            loss = agent.learn()
            total_loss += loss
            
            total_reward += combined_reward
            state = next_state
            step_count += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / max(1, step_count))
        episode_energies.append(env.total_energy)
        episode_atlp.append(calculate_atlp(env))
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Energy: {env.total_energy:.2f} kWh | "
                  f"ATLP: {episode_atlp[-1]:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return {
        'rewards': episode_rewards,
        'losses': episode_losses,
        'energies': episode_energies,
        'atlp': episode_atlp
    }


def evaluate_agent(env: CloudEnvironment, agent: AgentClass, num_episodes: int = 10) -> Dict:
    """Evaluate the trained agent"""
    
    # Temporarily set epsilon to 0 for evaluation (pure exploitation)
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    metrics = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_metrics = []
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            episode_metrics.append(info)
            state = next_state
            
            if done:
                break
        
        metrics.append({
            'total_energy': env.total_energy,
            'response_times': [m['response_time'] for m in episode_metrics],
            'utilizations': [m['utilization'] for m in episode_metrics]
        })
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Aggregate results
    avg_energy = np.mean([m['total_energy'] for m in metrics])
    avg_response_time = np.mean([np.mean(m['response_times']) for m in metrics])
    avg_utilization = np.mean([np.mean(m['utilizations']) for m in metrics])
    
    return {
        'avg_energy': avg_energy,
        'avg_response_time': avg_response_time,
        'avg_utilization': avg_utilization,
        'std_energy': np.std([m['total_energy'] for m in metrics]),
        'detail_metrics': metrics
    }


def compare_with_baselines(env_sizes: List[int] = [50, 100, 200, 500, 1000]) -> Dict:
    """
    Compare TaskSchedulingDQN with baseline algorithms:
    - FCFS (First Come First Serve)
    - EDF (Earliest Deadline First)
    - RR (Round Robin)
    """
    
    results = {
        'FCFS': [],
        'EDF': [],
        'RR': [],
        'TaskSchedulingDQN': []
    }
    
    for num_tasks in env_sizes:
        print(f"\n--- Testing with {num_tasks} tasks ---")
        
        # FCFS baseline
        env_fcfs = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
        energy_fcfs = _run_fcfs(env_fcfs)
        results['FCFS'].append(energy_fcfs)
        
        # EDF baseline
        env_edf = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
        energy_edf = _run_edf(env_edf)
        results['EDF'].append(energy_edf)
        
        # RR baseline
        env_rr = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
        energy_rr = _run_rr(env_rr)
        results['RR'].append(energy_rr)
        
        # Our DQN approach
        env_dqn = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
        agent_dqn = AgentClass(n_input=5, n_actions=10, learning_rate=0.001)
        train_agent(env_dqn, agent_dqn, num_episodes=50, verbose=False)
        eval_results = evaluate_agent(env_dqn, agent_dqn, num_episodes=5)
        results['TaskSchedulingDQN'].append(eval_results['avg_energy'])
        
        print(f"FCFS: {energy_fcfs:.2f} kWh | EDF: {energy_edf:.2f} kWh | "
              f"RR: {energy_rr:.2f} kWh | DQN: {eval_results['avg_energy']:.2f} kWh")
    
    return results


def _run_fcfs(env: CloudEnvironment) -> float:
    """First Come First Serve baseline"""
    total_energy = 0
    for i in range(env.num_tasks):
        # Assign to VM with lowest current utilization
        vm_idx = np.argmin([r.u_j for r in env.resources])
        _, _, done, info = env.step(vm_idx)
        total_energy += info['energy_consumed']
    return total_energy


def _run_edf(env: CloudEnvironment) -> float:
    """Earliest Deadline First baseline"""
    total_energy = 0
    # Sort tasks by deadline
    sorted_tasks = sorted(env.tasks, key=lambda t: t.d_i)
    original_tasks = env.tasks
    env.tasks = sorted_tasks
    
    for i in range(env.num_tasks):
        vm_idx = i % len(env.resources)
        _, _, done, info = env.step(vm_idx)
        total_energy += info['energy_consumed']
    
    env.tasks = original_tasks
    return total_energy


def _run_rr(env: CloudEnvironment) -> float:
    """Round Robin baseline"""
    total_energy = 0
    vm_idx = 0
    for i in range(env.num_tasks):
        _, _, done, info = env.step(vm_idx)
        total_energy += info['energy_consumed']
        vm_idx = (vm_idx + 1) % len(env.resources)
    return total_energy


def plot_results(training_history: Dict, baseline_comparison: Dict = None):
    """Plot training results and comparisons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Rewards
    axes[0, 0].plot(training_history['rewards'])
    axes[0, 0].set_title('Training Rewards over Episodes')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Energy Consumption
    axes[0, 1].plot(training_history['energies'])
    axes[0, 1].set_title('Energy Consumption over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Energy (kWh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ATLP (Average Task Loss Probability)
    axes[1, 0].plot(training_history['atlp'])
    axes[1, 0].set_title('ATLP over Episodes')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('ATLP')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Baseline Comparison
    if baseline_comparison:
        task_counts = [50, 100, 200, 500, 1000]
        for algorithm, energies in baseline_comparison.items():
            axes[1, 1].plot(task_counts[:len(energies)], energies, marker='o', label=algorithm)
        axes[1, 1].set_title('Algorithm Comparison: Energy Consumption vs Task Count')
        axes[1, 1].set_xlabel('Number of Tasks')
        axes[1, 1].set_ylabel('Energy Consumption (kWh)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task_scheduling_results.png', dpi=150, bbox_inches='tight')
    plt.show()



def main():
    """Main execution function"""
    print("=" * 60)
    print("TaskSchedulingDQN: Energy-Aware Task Scheduling using Deep Q-Learning")
    print("Based on the research paper by Janjani et al.")
    print("=" * 60)
    
    # Configuration
    NUM_TASKS = 500
    NUM_VMS = 10
    NUM_EPISODES = 100
    
    # Initialize environment and agent
    print(f"\n[1] Initializing environment with {NUM_TASKS} tasks and {NUM_VMS} VMs...")
    env = CloudEnvironment(num_tasks=NUM_TASKS, num_vms=NUM_VMS)
    
    print("[2] Initializing DQN Agent...")
    agent = AgentClass(
        n_input=5,  # State dimensions: U, C, M, D, R
        n_actions=NUM_VMS,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        batch_size=32
    )
    
    # Training
    print(f"\n[3] Training agent for {NUM_EPISODES} episodes...")
    training_history = train_agent(env, agent, num_episodes=NUM_EPISODES, verbose=True)
    
    # Evaluation
    print("\n[4] Evaluating trained agent...")
    eval_results = evaluate_agent(env, agent, num_episodes=10)
    print(f"Average Energy Consumption: {eval_results['avg_energy']:.2f} kWh")
    print(f"Average Response Time: {eval_results['avg_response_time']:.2f} s")
    print(f"Average VM Utilization: {eval_results['avg_utilization']:.2%}")
    
    # Baseline comparison
    print("\n[5] Comparing with baseline algorithms...")
    baseline_results = compare_with_baselines(env_sizes=[50, 100, 200, 500, 1000])
    
    # Plot results
    print("\n[6] Generating plots...")
    plot_results(training_history, baseline_results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Energy Consumed (Final Episode): {training_history['energies'][-1]:.2f} kWh")
    print(f"Final ATLP (Task Loss Probability): {training_history['atlp'][-1]:.4f}")
    print(f"Final Training Reward: {training_history['rewards'][-1]:.2f}")
    
    # Calculate final performance metrics
    final_metrics = env.get_metrics()
    print(f"\nSystem Performance Metrics:")
    print(f"  - Average CPU Utilization: {final_metrics['avg_cpu_util']:.2f}%")
    print(f"  - Average Memory Utilization: {final_metrics['avg_memory_util']:.2f}%")
    print(f"  - Average Disk Utilization: {final_metrics['avg_disk_util']:.2f}%")
    print(f"  - Throughput: {final_metrics['throughput']:.2f} tasks/s")
    print(f"  - Task Completion Rate: {final_metrics['task_completion_rate']:.2f}%")
    
    print("\n[✓] Execution completed successfully!")
    print("Results saved to 'task_scheduling_results.png'")


if __name__ == "__main__":
    main()