"""Hyperparameter optimization for TaskSchedulingDQN scheduler."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import torch

from agents.scheduler import SystemScheduler
from core.environment import CloudEnvironment


class HyperparameterTuner:
    """Systematic hyperparameter tuning for the DQN cloud scheduler."""

    def __init__(self, num_tasks: int = 100, num_vms: int = 10):
        self.num_tasks = num_tasks
        self.num_vms = num_vms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_config(
        self, params: Dict[str, Any], episodes: int = 20, seed: int = 42
    ) -> Dict[str, float]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = CloudEnvironment(num_tasks=self.num_tasks, num_vms=self.num_vms)
        scheduler = SystemScheduler(
            n_input=6,
            n_actions=self.num_vms,
            device=self.device,
            learning_rate=params.get("learning_rate", 0.001),
            gamma=params.get("gamma", 0.99),
            exploration_rate=1.0,
            exploration_min=params.get("exploration_min", 0.01),
            exploration_decay=params.get("exploration_decay", 0.995),
            batch_size=params.get("batch_size", 64),
        )

        rewards = []
        energies = []
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            while True:
                action = scheduler.select_action(state)
                next_state, base_reward, done, info = env.step(action)
                uptime, cpu_util, mem_util, disk_util, ram_util, load_balance = state
                dq1, r1 = scheduler.process_utilization_stage(state, action, next_state, uptime, mem_util, disk_util)
                dq2, r2 = scheduler.process_load_stage(state, action, next_state, cpu_util, ram_util)
                combined_gain = base_reward + 0.15 * (r1 + r2) + 0.1 * load_balance

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    current_val = scheduler.model(state_tensor)[0, action].item()
                    next_val_max = scheduler.target_model(next_state_tensor).max().item()
                    error = combined_gain + scheduler.gamma * next_val_max - current_val

                scheduler.store_transition(state, action, combined_gain, next_state, done, error)
                scheduler.update_model()
                episode_reward += combined_gain
                state = next_state
                if done:
                    break
            rewards.append(episode_reward)
            energies.append(env.total_energy)
            scheduler.exploration_rate = max(
                scheduler.exploration_min,
                scheduler.exploration_rate * scheduler.exploration_decay,
            )

        return {
            "avg_reward": float(np.mean(rewards)),
            "avg_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
        }

    def grid_search(
        self, param_grid: Dict[str, List[Any]], episodes: int = 15
    ) -> Dict[str, Any]:
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        print(f"Testing {len(combinations)} hyperparameter combinations...")
        best_params = None
        best_energy = float("inf")
        all_results = []

        for i, combination in enumerate(combinations, 1):
            params = dict(zip(param_names, combination))
            print(f"[{i}/{len(combinations)}] Evaluating: {params}")
            metrics = self.evaluate_config(params, episodes=episodes)
            result = {"params": params, **metrics}
            all_results.append(result)
            print(
                f"    -> Energy: {metrics['avg_energy']:.2f} kWh (±{metrics['std_energy']:.2f}) | Avg Reward: {metrics['avg_reward']:.2f}"
            )

            if metrics["avg_energy"] < best_energy:
                best_energy = metrics["avg_energy"]
                best_params = params
                print("    *** New best configuration found! ***")

        return {
            "best_params": best_params,
            "best_energy": best_energy,
            "all_results": all_results,
        }


def run_tuning():
    print("=" * 65)
    print(" Hyperparameter Tuning for Energy-Aware Task Scheduling DQN")
    print("=" * 65)
    tuner = HyperparameterTuner(num_tasks=200, num_vms=10)
    param_grid = {
        "learning_rate": [0.0005, 0.001],
        "gamma": [0.95, 0.99],
        "batch_size": [32, 64],
        "exploration_decay": [0.99, 0.995],
    }
    results = tuner.grid_search(param_grid, episodes=10)
    print("\n" + "=" * 65)
    print(" TUNING RESULTS SUMMARY")
    print("=" * 65)
    print(f"Best Hyperparameters: {results['best_params']}")
    print(f"Best Average Energy: {results['best_energy']:.2f} kWh")


if __name__ == "__main__":
    run_tuning()