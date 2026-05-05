import numpy as np
import random
from typing import List, Tuple, Dict
from core.models import Task, Resource

class CloudEnvironment:
    def __init__(self, num_tasks: int = 100, num_vms: int = 10):
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        self.tasks = self._initialize_tasks_realistic()
        self.resources = self._initialize_resources_heterogeneous()
        self.workload_distribution = np.zeros((num_tasks, num_vms))
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.total_energy = 0.0
        self.energy_trace = []
        self.metrics_history = {
            'energy': [], 'cpu_util': [], 'memory_util': [], 'disk_util': [],
            'response_times': [], 'uptime': [], 'throughput': [], 'load_balance': [],
            'task_completion': [], 'deadline_misses': [], 'energy_efficiency': []
        }
        
    def _initialize_tasks_realistic(self) -> List[Task]:
        tasks = []
        patterns = [
            {'r_range': (100, 300), 'd_range': (10, 30), 'e_range': (0.5, 1.5), 'prob': 0.3},
            {'r_range': (300, 700), 'd_range': (30, 60), 'e_range': (1.5, 3.0), 'prob': 0.5},
            {'r_range': (700, 1000), 'd_range': (60, 100), 'e_range': (3.0, 5.0), 'prob': 0.2}
        ]
        for i in range(self.num_tasks):
            pattern = np.random.choice(patterns, p=[p['prob'] for p in patterns])
            task = Task(
                task_id=i,
                r_i=random.uniform(*pattern['r_range']),
                d_i=random.uniform(*pattern['d_range']),
                e_i=random.uniform(*pattern['e_range']),
                p_i=np.random.exponential(scale=0.5),
                dep_i=random.sample(range(max(0, i-10), i), min(5, i)) if i > 0 and random.random() < 0.3 else []
            )
            tasks.append(task)
        tasks.sort(key=lambda x: (x.p_i, -x.d_i))
        return tasks
    
    def _initialize_resources_heterogeneous(self) -> List[Resource]:
        resources = []
        for i in range(self.num_vms):
            if i < self.num_vms // 3:
                capacity = random.uniform(1500, 2000)
                energy_rate = random.uniform(0.4, 0.5)
            elif i < 2 * self.num_vms // 3:
                capacity = random.uniform(800, 1500)
                energy_rate = random.uniform(0.25, 0.4)
            else:
                capacity = random.uniform(500, 800)
                energy_rate = random.uniform(0.1, 0.25)
            resource = Resource(
                vm_id=i,
                c_j=capacity,
                eta_j=energy_rate,
                u_j=0.0
            )
            resources.append(resource)
        return resources
    
    def get_state(self) -> np.ndarray:
        uptime = min(100, self.current_time / max(1, self.num_tasks) * 100)
        cpu_util = self._get_cpu_utilization()
        memory_util = self._get_memory_utilization()
        disk_util = self._get_disk_utilization()
        ram_util = memory_util
        load_balance = self._get_load_balance()
        return np.array([uptime, cpu_util, memory_util, disk_util, ram_util, load_balance])
    
    def _get_cpu_utilization(self) -> float:
        if not self.resources:
            return 0.0
        weights = np.array([r.c_j for r in self.resources])
        weights = weights / weights.sum()
        total_util = sum(r.u_j * w for r, w in zip(self.resources, weights))
        return total_util * 100
    
    def _get_memory_utilization(self) -> float:
        base_util = 20 + 60 * (self.current_task_idx / max(1, self.num_tasks))
        fragmentation = 5 * np.sin(self.current_task_idx / 10)
        return min(95, max(10, base_util + fragmentation))
    
    def _get_disk_utilization(self) -> float:
        io_intensity = 0.3 * (self.current_task_idx / max(1, self.num_tasks))
        base_util = 10 + 50 * (self.current_task_idx / max(1, self.num_tasks)) + io_intensity
        return min(85, base_util)
    
    def _get_load_balance(self) -> float:
        if not self.resources:
            return 1.0
        utilizations = [r.u_j for r in self.resources]
        variance = np.var(utilizations) if len(utilizations) > 1 else 0
        return 1.0 / (1.0 + variance)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0.0, True, {}
        current_task = self.tasks[self.current_task_idx]
        selected_vm = self.resources[action] if action < len(self.resources) else self.resources[0]
        base_exec_time = current_task.r_i / selected_vm.c_j
        contention_factor = 1 + selected_vm.u_j * 0.5
        execution_time = base_exec_time * contention_factor
        base_energy = selected_vm.eta_j * current_task.r_i * current_task.d_i / 3600
        energy_scaling = 1 + 0.2 * selected_vm.u_j
        task_energy = base_energy * energy_scaling
        self.total_energy += task_energy
        self.energy_trace.append(task_energy)
        utilization_increment = (current_task.r_i / selected_vm.c_j) * 0.1
        selected_vm.u_j = min(1.0, selected_vm.u_j * 0.95 + utilization_increment)
        self.workload_distribution[self.current_task_idx, action] = 1
        response_time = execution_time
        deadline_met = response_time <= current_task.d_i
        self.metrics_history['response_times'].append(response_time)
        self.metrics_history['deadline_misses'].append(0 if deadline_met else 1)
        self.metrics_history['energy'].append(self.total_energy)
        self.metrics_history['cpu_util'].append(self._get_cpu_utilization())
        self.metrics_history['memory_util'].append(self._get_memory_utilization())
        self.metrics_history['disk_util'].append(self._get_disk_utilization())
        self.metrics_history['uptime'].append(self.current_time)
        self.metrics_history['load_balance'].append(self._get_load_balance())
        energy_reward = -task_energy * 10
        deadline_reward = 10 if deadline_met else -20
        utilization_reward = -abs(selected_vm.u_j - 0.7) * 5
        load_balance_reward = self._get_load_balance() * 5
        reward = energy_reward + deadline_reward + utilization_reward + load_balance_reward
        self.current_task_idx += 1
        self.current_time += execution_time
        self.metrics_history['task_completion'].append(self.current_task_idx)
        if self.current_time > 0:
            throughput = self.current_task_idx / self.current_time
            self.metrics_history['throughput'].append(throughput)
        done = self.current_task_idx >= len(self.tasks)
        next_state = self.get_state()
        info = {
            'task_id': current_task.task_id,
            'vm_id': selected_vm.vm_id,
            'energy_consumed': task_energy,
            'response_time': response_time,
            'utilization': selected_vm.u_j,
            'deadline_met': deadline_met
        }
        return next_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        self.current_time = 0.0
        self.current_task_idx = 0
        self.completed_tasks = []
        self.total_energy = 0.0
        self.energy_trace = []
        self.workload_distribution = np.zeros((self.num_tasks, self.num_vms))
        for resource in self.resources:
            resource.u_j = 0.0
        for key in self.metrics_history:
            self.metrics_history[key] = []
        return self.get_state()
    
    def get_metrics(self) -> Dict:
        return {
            'total_energy': self.total_energy,
            'avg_cpu_util': self._get_cpu_utilization(),
            'avg_memory_util': self._get_memory_utilization(),
            'avg_disk_util': self._get_disk_utilization(),
            'avg_response_time': np.mean(self.metrics_history['response_times']) if self.metrics_history['response_times'] else 0,
            'throughput': self.metrics_history['throughput'][-1] if self.metrics_history['throughput'] else 0,
            'task_completion_rate': self.current_task_idx / max(1, self.num_tasks) * 100,
            'deadline_met_rate': 1 - np.mean(self.metrics_history['deadline_misses']) if self.metrics_history['deadline_misses'] else 1,
            'avg_load_balance': np.mean(self.metrics_history['load_balance']) if self.metrics_history['load_balance'] else 0,
            'energy_efficiency': self.current_task_idx / max(1, self.total_energy) if self.total_energy > 0 else 0
        }
