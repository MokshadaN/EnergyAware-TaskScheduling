from dataclasses import dataclass, field
from typing import List

@dataclass
class Task:
    task_id: int
    r_i: float
    d_i: float
    e_i: float
    p_i: float
    dep_i: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.dep_i is None:
            self.dep_i = []

@dataclass
class Resource:
    vm_id: int
    c_j: float
    eta_j: float
    u_j: float
