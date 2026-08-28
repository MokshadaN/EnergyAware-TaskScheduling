"""Interactive dashboard for comparing cloud task scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass

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
    return pd.DataFrame({"task": np.arange(1, task_count + 1), "demand": demand, "deadline": deadline})


def build_cluster(vm_count: int, seed: int) -> list[VirtualMachine]:
    rng = np.random.default_rng(seed + 1)
    capacities = rng.uniform(550, 1900, vm_count)
    rates = np.interp(capacities, (capacities.min(), capacities.max()), (0.12, 0.48))
    return [VirtualMachine(f"VM-{i + 1}", capacity, rate) for i, (capacity, rate) in enumerate(zip(capacities, rates))]


def choose_vm(policy: str, machines: list[VirtualMachine], task_number: int) -> int:
    if policy == "Round robin":
        return task_number % len(machines)
    if policy == "Least utilized":
        return int(np.argmin([vm.utilization for vm in machines]))
    # A transparent multi-objective score: energy, current load and available capacity.
    scores = [vm.energy_rate * 0.55 + vm.utilization * 0.30 + (1 - vm.capacity / 2000) * 0.15 for vm in machines]
    return int(np.argmin(scores))


def run_simulation(workload: pd.DataFrame, vm_count: int, policy: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    machines = build_cluster(vm_count, seed)
    records = []
    for index, task in workload.iterrows():
        vm = machines[choose_vm(policy, machines, index)]
        execution_time = (task.demand / vm.capacity) * (1 + 0.45 * vm.utilization)
        energy = vm.energy_rate * task.demand * execution_time / 3600
        vm.utilization = min(1.0, vm.utilization * 0.90 + task.demand / vm.capacity * 0.10)
        records.append({
            "task": int(task.task), "vm": vm.name, "energy_kwh": energy,
            "response_time_s": execution_time, "deadline_met": execution_time <= task.deadline,
            "utilization": vm.utilization,
        })
    results = pd.DataFrame(records)
    summary = results.groupby("vm", as_index=False).agg(
        tasks=("task", "count"), energy_kwh=("energy_kwh", "sum"), utilization=("utilization", "last")
    )
    return results, summary


st.set_page_config(page_title="Energy-Aware Scheduler", page_icon="⚡", layout="wide")
st.title("Energy-Aware Cloud Scheduler")
st.caption("Explore how task-placement policies trade energy use, latency, and load distribution in a simulated cloud cluster.")

with st.sidebar:
    st.header("Scenario")
    task_count = st.slider("Tasks", 25, 1_000, 250, 25)
    vm_count = st.slider("Virtual machines", 3, 30, 10)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    policy = st.selectbox("Policy", ["Energy-aware", "Least utilized", "Round robin"])

workload = build_workload(task_count, int(seed))
results, by_vm = run_simulation(workload, vm_count, policy, int(seed))
total_energy = results.energy_kwh.sum()
deadline_rate = results.deadline_met.mean()
avg_response = results.response_time_s.mean()

a, b, c, d = st.columns(4)
a.metric("Energy consumed", f"{total_energy:.3f} kWh")
b.metric("Deadline success", f"{deadline_rate:.1%}")
c.metric("Mean response time", f"{avg_response:.2f} s")
d.metric("Tasks completed", f"{len(results):,}")

left, right = st.columns(2)
with left:
    st.subheader("Energy by virtual machine")
    st.plotly_chart(px.bar(by_vm, x="vm", y="energy_kwh", color="energy_kwh", color_continuous_scale="Blues", labels={"energy_kwh": "kWh", "vm": "Virtual machine"}), use_container_width=True)
with right:
    st.subheader("Utilization profile")
    st.plotly_chart(px.bar(by_vm, x="vm", y="utilization", color="utilization", range_y=[0, 1], color_continuous_scale="Teal", labels={"utilization": "Utilization", "vm": "Virtual machine"}), use_container_width=True)

st.subheader("Scheduling timeline")
timeline = results.melt(id_vars="task", value_vars=["energy_kwh", "response_time_s"], var_name="metric", value_name="value")
st.plotly_chart(px.line(timeline, x="task", y="value", color="metric", labels={"task": "Task", "value": "Value"}), use_container_width=True)

with st.expander("About this simulation"):
    st.write("The dashboard uses a deterministic synthetic workload and a transparent heuristic score for the energy-aware policy. It is intended for comparison and demonstration, not for reporting real data-center measurements.")

st.download_button("Download run data (CSV)", results.to_csv(index=False).encode("utf-8"), "scheduling-run.csv", "text/csv")
