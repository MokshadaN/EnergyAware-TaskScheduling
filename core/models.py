"""Data models for cloud tasks and virtual machine resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Task:
    """Represents a computational task with resource demands and deadline constraints."""

    task_id: int
    r_i: float  # Resource demand / execution requirement (MIPS)
    d_i: float  # Deadline constraint (seconds)
    e_i: float  # Energy baseline requirement (kWh)
    p_i: float  # Priority rating
    dep_i: List[int] = field(default_factory=list)  # Task dependencies (predecessor task IDs)

    def __post_init__(self) -> None:
        if self.dep_i is None:
            self.dep_i = []


@dataclass
class Resource:
    """Represents a Virtual Machine (VM) resource in the cloud cluster."""

    vm_id: int
    c_j: float  # Computing capacity (MIPS)
    eta_j: float  # Energy rate coefficient (kW/MIPS)
    u_j: float = 0.0  # Current resource utilization [0.0, 1.0]

    def reset(self) -> None:
        self.u_j = 0.0
