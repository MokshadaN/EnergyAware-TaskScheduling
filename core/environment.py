import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict

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
        self.tasks = self._init_tasks()
        self.resources = self._init_resources()
        self.reset()

    def _init_tasks(self):
        return [Task(
            task_id=i,
            r_i=random.uniform(100, 1000),
            d_i=random.uniform(10, 100),
            e_i=random.uniform(0.5, 5.0),
            p_i=random.uniform(0, 1),
            dep_i=random.sample(range(max(0, i-5), i), min(3, i)) if i > 0 else []
        ) for i in range(self.num_tasks)]

    def _init_resources(self):
        return [Resource(i, random.uniform(500, 2000), random.uniform(0.1, 0.5), 0.0) 
                for i in range(self.num_vms)]

    def reset(self):
        self.current_time = 0.0
        self.current_task_idx = 0
        self.total_energy = 0.0
        self.failed_tasks = []
        for r in self.resources: r.u_j = 0.0
        return self.get_state()

    def get_state(self):
        uptime = self.current_time / 100.0
        cpu = self._cpu_util() / 100.0
        mem = self._mem_util() / 100.0
        disk = self._disk_util() / 100.0
        return np.array([uptime, cpu, mem, disk, mem])

    def _cpu_util(self):
        return sum(r.u_j for r in self.resources) / len(self.resources) * 100

    def _mem_util(self):
        return min(95, 20 + 70 * (self.current_task_idx / self.num_tasks))

    def _disk_util(self):
        return min(85, 10 + 60 * (self.current_task_idx / self.num_tasks))

    def step(self, action):
        if self.current_task_idx >= len(self.tasks):
            return self.get_state(), 0.0, True, {}
        
        task = self.tasks[self.current_task_idx]
        vm = self.resources[action]
        
        exec_time = task.r_i / vm.c_j
        energy = vm.eta_j * task.r_i * exec_time
        
        success = exec_time <= task.d_i
        if not success:
            self.failed_tasks.append(task.task_id)
            reward = -10.0
        else:
            self.total_energy += energy
            vm.u_j = min(1.0, vm.u_j + (task.r_i / vm.c_j) * 0.1)
            reward = 1.0 - (energy / 10.0)
            
        self.current_task_idx += 1
        self.current_time += exec_time
        
        return self.get_state(), reward, self.current_task_idx >= len(self.tasks), {
            'energy_consumed': energy,
            'success': success,
            'response_time': exec_time,
            'utilization': vm.u_j
        }

    def get_metrics(self):
        return {
            'avg_cpu_util': self._cpu_util(),
            'avg_memory_util': self._mem_util(),
            'avg_disk_util': self._disk_util(),
            'throughput': self.current_task_idx / max(1, self.current_time),
            'task_completion_rate': (self.current_task_idx - len(self.failed_tasks)) / self.num_tasks * 100
        }

def calculate_atlp(env):
    return len(env.failed_tasks) / max(1, env.num_tasks)
