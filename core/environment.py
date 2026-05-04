import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Task:
    task_id: int
    r_i: float  # MIPS requirement
    d_i: float  # Deadline in seconds
    e_i: float  # Energy in kWh
    p_i: float  # Priority
    dep_i: List[int]
    
    def __post_init__(self):
        if self.dep_i is None:
            self.dep_i = []

@dataclass
class Resource:
    vm_id: int
    c_j: float  # Capacity in MIPS
    eta_j: float  # Energy rate in kW
    u_j: float  # Current utilization

class CloudEnvironment:
    def __init__(self, num_tasks=100, num_vms=10):
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        self.tasks = self._init_tasks_realistic()
        self.resources = self._init_resources_heterogeneous()
        self.reset()

    def _init_tasks_realistic(self):
        tasks = []
        patterns = [
            {'r_range': (100, 300), 'd_range': (10, 30), 'e_range': (0.5, 1.5), 'prob': 0.3},
            {'r_range': (300, 700), 'd_range': (30, 60), 'e_range': (1.5, 3.0), 'prob': 0.5},
            {'r_range': (700, 1000), 'd_range': (60, 100), 'e_range': (3.0, 5.0), 'prob': 0.2}
        ]
        for i in range(self.num_tasks):
            p = np.random.choice(patterns, p=[x['prob'] for x in patterns])
            tasks.append(Task(
                task_id=i,
                r_i=random.uniform(*p['r_range']),
                d_i=random.uniform(*p['d_range']),
                e_i=random.uniform(*p['e_range']),
                p_i=np.random.exponential(0.5),
                dep_i=random.sample(range(max(0, i-10), i), min(5, i)) if i > 0 and random.random() < 0.3 else []
            ))
        tasks.sort(key=lambda x: (x.p_i, -x.d_i))
        return tasks

    def _init_resources_heterogeneous(self):
        resources = []
        for i in range(self.num_vms):
            if i < self.num_vms // 3:
                c, eta = random.uniform(1500, 2000), random.uniform(0.4, 0.5)
            elif i < 2 * self.num_vms // 3:
                c, eta = random.uniform(800, 1500), random.uniform(0.25, 0.4)
            else:
                c, eta = random.uniform(500, 800), random.uniform(0.1, 0.25)
            resources.append(Resource(i, c, eta, 0.0))
        return resources

    def reset(self):
        self.current_time = 0.0
        self.current_task_idx = 0
        self.total_energy = 0.0
        for r in self.resources: r.u_j = 0.0
        self.metrics_history = {k: [] for k in ['energy', 'cpu_util', 'memory_util', 'disk_util', 
                                              'response_times', 'uptime', 'throughput', 'load_balance', 
                                              'task_completion', 'deadline_misses']}
        return self.get_state()

    def get_state(self):
        uptime = min(100, self.current_time / max(1, self.num_tasks) * 100)
        cpu = self._cpu_util()
        mem = self._mem_util()
        disk = self._disk_util()
        lb = self._load_balance()
        return np.array([uptime, cpu, mem, disk, mem, lb])

    def _cpu_util(self):
        weights = np.array([r.c_j for r in self.resources])
        weights /= weights.sum()
        return sum(r.u_j * w for r, w in zip(self.resources, weights)) * 100

    def _mem_util(self):
        base = 20 + 60 * (self.current_task_idx / max(1, self.num_tasks))
        frag = 5 * np.sin(self.current_task_idx / 10)
        return min(95, max(10, base + frag))

    def _disk_util(self):
        return min(85, 10 + 50 * (self.current_task_idx / max(1, self.num_tasks)))

    def _load_balance(self):
        utils = [r.u_j for r in self.resources]
        return 1.0 / (1.0 + np.var(utils)) if len(utils) > 1 else 1.0

    def step(self, action):
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0.0, True, {}
        
        task = self.tasks[self.current_task_idx]
        vm = self.resources[action]
        
        exec_time = (task.r_i / vm.c_j) * (1 + vm.u_j * 0.5)
        energy = (vm.eta_j * task.r_i * task.d_i / 3600) * (1 + 0.2 * vm.u_j)
        
        vm.u_j = min(1.0, vm.u_j * 0.95 + (task.r_i / vm.c_j) * 0.1)
        self.total_energy += energy
        
        met = exec_time <= task.d_i
        reward = (-energy * 10) + (10 if met else -20) + (-abs(vm.u_j - 0.7) * 5) + (self._load_balance() * 5)
        
        self.current_task_idx += 1
        self.current_time += exec_time
        
        self.metrics_history['response_times'].append(exec_time)
        self.metrics_history['deadline_misses'].append(0 if met else 1)
        self.metrics_history['load_balance'].append(self._load_balance())
        if self.current_time > 0:
            self.metrics_history['throughput'].append(self.current_task_idx / self.current_time)
            
        return self.get_state(), reward, self.current_task_idx >= len(self.tasks), {
            'energy_consumed': energy, 'response_time': exec_time, 'utilization': vm.u_j, 'deadline_met': met
        }
