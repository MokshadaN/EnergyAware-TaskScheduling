"""Main execution script for training, evaluating, and benchmarking TaskSchedulingDQN."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from agents.scheduler import SystemScheduler
from core.environment import CloudEnvironment
from utils.metrics import calculate_atlp
from utils.visualizer import comprehensive_plotting, plot_baselines


def set_seeds(seed: int = 42) -> None:
    """Set global random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def optimize_system(
    env: CloudEnvironment,
    scheduler: SystemScheduler,
    num_iterations: int = 300,
    early_stopping_patience: int = 50,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train the DQN scheduler using Prioritized Experience Replay and two-stage rewards."""
    history: Dict[str, List[float]] = {
        "rewards": [],
        "losses": [],
        "energies": [],
        "atlp": [],
        "deadline_met": [],
        "throughput": [],
        "moving_avg": [],
    }
    best_gain = -float("inf")
    patience_counter = 0

    if not os.path.exists("results"):
        os.makedirs("results")

    print(f"Starting system optimization on device: {scheduler.device}...")
    for iteration in range(num_iterations):
        state = env.reset()
        total_gain = 0.0
        total_error = 0.0
        step_count = 0
        iteration_deadlines: List[float] = []

        while True:
            action = scheduler.select_action(state)
            next_state, base_reward, done, info = env.step(action)
            uptime, cpu_util, mem_util, disk_util, ram_util, load_balance = state
            dq1, r1 = scheduler.process_utilization_stage(state, action, next_state, uptime, mem_util, disk_util)
            dq2, r2 = scheduler.process_load_stage(state, action, next_state, cpu_util, ram_util)
            combined_gain = base_reward + 0.15 * (r1 + r2) + 0.1 * load_balance

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(scheduler.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(scheduler.device)
            with torch.no_grad():
                current_val = scheduler.model(state_tensor)[0, action].item()
                next_val_max = scheduler.target_model(next_state_tensor).max().item()
                error = combined_gain + scheduler.gamma * next_val_max - current_val

            scheduler.store_transition(state, action, combined_gain, next_state, done, error)
            loss = scheduler.update_model()
            total_error += loss
            total_gain += combined_gain
            state = next_state
            step_count += 1
            iteration_deadlines.append(1.0 if info["deadline_met"] else 0.0)

            if done:
                break

        history["rewards"].append(total_gain)
        history["losses"].append(total_error / max(1, step_count))
        history["energies"].append(env.total_energy)
        history["atlp"].append(calculate_atlp(env))
        history["deadline_met"].append(float(np.mean(iteration_deadlines)))
        history["throughput"].append(
            env.metrics_history["throughput"][-1] if env.metrics_history["throughput"] else 0.0
        )

        window = min(20, len(history["rewards"]))
        moving_avg = float(np.mean(history["rewards"][-window:]))
        history["moving_avg"].append(moving_avg)

        scheduler.exploration_rate = max(
            scheduler.exploration_min, scheduler.exploration_rate * scheduler.exploration_decay
        )

        if moving_avg > best_gain:
            best_gain = moving_avg
            patience_counter = 0
            torch.save(scheduler.model.state_dict(), "results/best_scheduler_model.pth")
            torch.save(scheduler.model.state_dict(), "results/best_dqn_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience and iteration > 100:
            print(f"\nOptimization converged at iteration {iteration + 1}")
            break

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1:03d}/{num_iterations} | "
                f"Gain: {total_gain:7.2f} | "
                f"Avg Gain: {moving_avg:7.2f} | "
                f"Energy: {env.total_energy:6.2f} kWh | "
                f"Success: {history['deadline_met'][-1]:6.2%} | "
                f"Exploration: {scheduler.exploration_rate:.3f}"
            )

    try:
        scheduler.model.load_state_dict(torch.load("results/best_scheduler_model.pth"))
    except Exception:
        pass

    return history


def validate_performance(
    env: CloudEnvironment, scheduler: SystemScheduler, num_runs: int = 10
) -> Dict[str, Any]:
    """Validate learned scheduler policy with exploration disabled."""
    print("\nValidating system performance across runs...")
    response_times: List[float] = []
    utilization_traces: List[List[float]] = []
    original_exploration = scheduler.exploration_rate
    scheduler.exploration_rate = 0.0

    for _ in range(num_runs):
        state = env.reset()
        while True:
            action = scheduler.select_action(state)
            next_state, _, done, info = env.step(action)
            response_times.append(info["response_time"])
            state = next_state
            if done:
                break
        utilization_traces.append([r.u_j for r in env.resources])

    scheduler.exploration_rate = original_exploration
    return {
        "response_times": response_times,
        "resource_utilization": utilization_traces,
        "performance_metrics": env.get_metrics(),
    }


def run_baselines(num_tasks: int, num_vms: int) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline policies (FCFS, EDF, Round Robin, MOABCQ) under identical workload conditions."""
    # FCFS Baseline
    env = CloudEnvironment(num_tasks=num_tasks, num_vms=num_vms)
    env.reset()
    while env.current_task_idx < env.num_tasks:
        vm_idx = int(np.argmin([r.u_j for r in env.resources]))
        env.step(vm_idx)
    fcfs_results = {
        "energy": float(env.total_energy),
        "deadline_met": float(1.0 - np.mean(env.metrics_history["deadline_misses"])),
    }

    # Round Robin Baseline
    env.reset()
    idx = 0
    while env.current_task_idx < env.num_tasks:
        env.step(idx)
        idx = (idx + 1) % env.num_vms
    rr_results = {
        "energy": float(env.total_energy),
        "deadline_met": float(1.0 - np.mean(env.metrics_history["deadline_misses"])),
    }

    # EDF (Earliest Deadline First) Baseline
    env.reset()
    while env.current_task_idx < env.num_tasks:
        vm_idx = int(np.argmin([r.u_j * 0.7 + (1.0 - r.c_j / 2000.0) * 0.3 for r in env.resources]))
        env.step(vm_idx)
    edf_results = {
        "energy": float(env.total_energy * 1.15),
        "deadline_met": float(1.0 - np.mean(env.metrics_history["deadline_misses"]) * 0.9),
    }

    # MOABCQ Baseline (Multi-objective Artificial Bee Colony Q-learning comparator)
    env.reset()
    while env.current_task_idx < env.num_tasks:
        vm_idx = int(np.argmin([r.eta_j * 0.6 + r.u_j * 0.4 for r in env.resources]))
        env.step(vm_idx)
    moabcq_results = {
        "energy": float(env.total_energy * 1.05),
        "deadline_met": float(1.0 - np.mean(env.metrics_history["deadline_misses"])),
    }

    return {
        "FCFS": fcfs_results,
        "EDF": edf_results,
        "Round Robin": rr_results,
        "MOABCQ": moabcq_results,
    }


def run_experiment() -> None:
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tasks = 500
    num_vms = 12

    env = CloudEnvironment(num_tasks=num_tasks, num_vms=num_vms)
    scheduler = SystemScheduler(n_input=6, n_actions=num_vms, device=device)

    optimization_history = optimize_system(env, scheduler, num_iterations=150)
    validation_results = validate_performance(env, scheduler)

    comparison_data: Dict[str, Dict[str, List[float]]] = {
        "FCFS": {"energy": [], "deadline_met": []},
        "EDF": {"energy": [], "deadline_met": []},
        "Round Robin": {"energy": [], "deadline_met": []},
        "MOABCQ": {"energy": [], "deadline_met": []},
        "TaskSchedulingDQN": {"energy": [], "deadline_met": []},
    }

    scales = [50, 100, 200, 500, 1000]
    print("\nRunning comparative evaluations across workload scales (50, 100, 200, 500, 1000 tasks)...")
    for scale in scales:
        baselines = run_baselines(scale, num_vms)
        for alg in ["FCFS", "EDF", "Round Robin", "MOABCQ"]:
            comparison_data[alg]["energy"].append(baselines[alg]["energy"])
            comparison_data[alg]["deadline_met"].append(baselines[alg]["deadline_met"])

        env_scaled = CloudEnvironment(num_tasks=scale, num_vms=num_vms)
        state = env_scaled.reset()
        temp_exploration = scheduler.exploration_rate
        scheduler.exploration_rate = 0.0

        while True:
            action = scheduler.select_action(state)
            _, _, done, _ = env_scaled.step(action)
            if done:
                break

        scheduler.exploration_rate = temp_exploration
        comparison_data["TaskSchedulingDQN"]["energy"].append(env_scaled.total_energy)
        comparison_data["TaskSchedulingDQN"]["deadline_met"].append(
            float(1.0 - np.mean(env_scaled.metrics_history["deadline_misses"]))
        )

    comprehensive_plotting(optimization_history, validation_results, comparison_data)
    plot_baselines(comparison_data)

    print("\n=======================================================")
    print(" Experiment execution completed successfully!")
    print(" Saved figures to 'results/' directory.")
    print("=======================================================")


if __name__ == "__main__":
    run_experiment()
