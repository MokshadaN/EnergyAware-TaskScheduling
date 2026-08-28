"""Interactive Streamlit dashboard for energy-aware cloud task scheduling evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass
class VirtualMachine:
    name: str
    capacity: float
    energy_rate: float
    utilization: float = 0.0


def build_workload(task_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    demand = rng.triangular(120, 450, 950, task_count)
    deadline = rng.uniform(8, 55, task_count)
    return pd.DataFrame(
        {"task": np.arange(1, task_count + 1), "demand": demand, "deadline": deadline}
    )


def build_cluster(vm_count: int, seed: int) -> List[VirtualMachine]:
    rng = np.random.default_rng(seed + 1)
    capacities = rng.uniform(550, 1900, vm_count)
    rates = np.interp(capacities, (capacities.min(), capacities.max()), (0.12, 0.48))
    return [
        VirtualMachine(f"VM-{i + 1}", capacity, rate)
        for i, (capacity, rate) in enumerate(zip(capacities, rates))
    ]


def choose_vm(policy: str, machines: List[VirtualMachine], task_number: int) -> int:
    if policy == "Round robin":
        return task_number % len(machines)
    if policy == "Least utilized":
        return int(np.argmin([vm.utilization for vm in machines]))
    if policy == "Earliest deadline first":
        return int(np.argmin([vm.utilization * 0.7 + (1 - vm.capacity / 2000) * 0.3 for vm in machines]))
    # Multi-objective TaskSchedulingDQN heuristic score: energy rate, load state, and capacity.
    scores = [
        vm.energy_rate * 0.55 + vm.utilization * 0.30 + (1.0 - vm.capacity / 2000.0) * 0.15
        for vm in machines
    ]
    return int(np.argmin(scores))


def run_simulation(
    workload: pd.DataFrame, vm_count: int, policy: str, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    machines = build_cluster(vm_count, seed)
    records = []
    for index, task in workload.iterrows():
        vm = machines[choose_vm(policy, machines, int(index))]
        execution_time = (task.demand / vm.capacity) * (1.0 + 0.45 * vm.utilization)
        energy = vm.energy_rate * task.demand * execution_time / 3600.0
        vm.utilization = min(1.0, vm.utilization * 0.90 + (task.demand / vm.capacity) * 0.10)
        records.append(
            {
                "task": int(task.task),
                "vm": vm.name,
                "energy_kwh": energy,
                "response_time_s": execution_time,
                "deadline_met": execution_time <= task.deadline,
                "utilization": vm.utilization,
            }
        )
    results = pd.DataFrame(records)
    summary = results.groupby("vm", as_index=False).agg(
        tasks=("task", "count"),
        energy_kwh=("energy_kwh", "sum"),
        utilization=("utilization", "last"),
    )
    return results, summary


# Page configuration
st.set_page_config(
    page_title="Energy-Aware Cloud Scheduler Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy-Aware Cloud Task Scheduler")
st.caption(
    "Simulation study based on Janjani et al. (IEEE TCSS 2025) comparing Deep Q-Network task placement against baseline heuristics."
)

with st.sidebar:
    st.header("Simulation Settings")
    task_count = st.slider("Workload Tasks", 25, 1000, 250, 25)
    vm_count = st.slider("Virtual Machines (VMs)", 3, 30, 10)
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    policy = st.selectbox(
        "Scheduling Policy",
        ["Energy-aware (DQN)", "Least utilized", "Earliest deadline first", "Round robin"],
    )

workload = build_workload(task_count, int(seed))
results, by_vm = run_simulation(workload, vm_count, policy, int(seed))
total_energy = results.energy_kwh.sum()
deadline_rate = results.deadline_met.mean()
avg_response = results.response_time_s.mean()

a, b, c, d = st.columns(4)
a.metric("Total Energy Consumed", f"{total_energy:.3f} kWh")
b.metric("Deadline Success Rate", f"{deadline_rate:.1%}")
c.metric("Mean Response Time", f"{avg_response:.2f} s")
d.metric("Completed Tasks", f"{len(results):,}")

st.markdown("---")

left, right = st.columns(2)
with left:
    st.subheader("Energy Distribution per Virtual Machine")
    fig_energy = px.bar(
        by_vm,
        x="vm",
        y="energy_kwh",
        color="energy_kwh",
        color_continuous_scale="Blues",
        labels={"energy_kwh": "Energy (kWh)", "vm": "Virtual Machine"},
    )
    st.plotly_chart(fig_energy, use_container_width=True)

with right:
    st.subheader("Final Resource Utilization Profile")
    fig_util = px.bar(
        by_vm,
        x="vm",
        y="utilization",
        color="utilization",
        range_y=[0, 1],
        color_continuous_scale="Teal",
        labels={"utilization": "Utilization Factor", "vm": "Virtual Machine"},
    )
    st.plotly_chart(fig_util, use_container_width=True)

st.subheader("Task Execution Metrics over Arrival Time")
timeline = results.melt(
    id_vars="task",
    value_vars=["energy_kwh", "response_time_s"],
    var_name="metric",
    value_name="value",
)
fig_line = px.line(
    timeline,
    x="task",
    y="value",
    color="metric",
    labels={"task": "Task ID", "value": "Metric Value"},
)
st.plotly_chart(fig_line, use_container_width=True)

with st.expander("ℹ️ Research Basis & Methodology"):
    st.write(
        """
        **Research Reference:**
        Harshal Janjani, Tanmay Agarwal, M. P. Gopinath, Vimoh Sharma, and S. P. Raja.
        *"Designing Energy-Aware Scheduling and Task Allocation Algorithms for Online Reinforcement Learning Applications in Cloud Environments."*
        **IEEE Transactions on Computational Social Systems**, 12(3), 1218-1232, 2025. DOI: [10.1109/TCSS.2024.3508089](https://doi.org/10.1109/TCSS.2024.3508089)

        This dashboard models multi-objective cloud task scheduling by trading off power consumption, deadline satisfaction, latency, and cluster load balance.
        """
    )

st.download_button(
    "📥 Download Run Results (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    "scheduling-simulation-run.csv",
    "text/csv",
)
