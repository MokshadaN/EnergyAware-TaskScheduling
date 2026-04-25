"""Visualization module for rendering system performance dashboards and baseline comparisons."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def comprehensive_plotting(
    training_history: Dict[str, Any],
    eval_results: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate and save the comprehensive multi-panel system performance dashboard."""
    if not os.path.exists("results"):
        os.makedirs("results")

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    fig = plt.figure(figsize=(20, 16))

    # Panel 1: Optimization Progress
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(training_history["rewards"], alpha=0.3, label="Iteration Gain", linewidth=0.8)
    ax1.plot(training_history["moving_avg"], "r-", label="Moving Average", linewidth=2)
    ax1.set_title("Optimization Progress", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Gain")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Energy Profile
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(training_history["energies"], "g-", linewidth=1.5)
    ax2.fill_between(range(len(training_history["energies"])), training_history["energies"], alpha=0.3)
    ax2.set_title("Energy Consumption Profile", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy (kWh)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Task Success Rate
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(training_history["deadline_met"], "m-", linewidth=1.5)
    ax3.set_title("Task Success Rate (Deadline Met)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Success Rate")
    ax3.set_ylim([0, 1.05])
    ax3.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="Target (95%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: System Throughput
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(training_history["throughput"], "c-", linewidth=1.5)
    ax4.set_title("System Throughput", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Tasks/Second")
    ax4.grid(True, alpha=0.3)

    # Panel 5: Task Loss Probability (ATLP)
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(training_history["atlp"], color="orange", linewidth=1.5)
    ax5.fill_between(range(len(training_history["atlp"])), training_history["atlp"], alpha=0.3)
    ax5.set_title("Task Loss Probability (ATLP)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Probability")
    ax5.grid(True, alpha=0.3)

    # Panel 6: Convergence Loss
    ax6 = plt.subplot(3, 4, 6)
    losses = [max(1e-6, loss) for loss in training_history["losses"]]
    ax6.plot(losses, color="brown", linewidth=1, alpha=0.7)
    ax6.set_title("Optimization Convergence (Loss)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("Error Value")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)

    # Panel 7: Energy vs Success Trade-off
    ax7 = plt.subplot(3, 4, 7)
    step = max(1, len(training_history["energies"]) // 50)
    energy_subset = training_history["energies"][::step]
    deadline_subset = training_history["deadline_met"][::step]
    colors = range(len(energy_subset))
    sc = ax7.scatter(energy_subset, deadline_subset, c=colors, cmap="viridis", alpha=0.8, s=50)
    ax7.set_title("Energy vs Success Trade-off", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Energy (kWh)")
    ax7.set_ylabel("Success Rate")
    if len(energy_subset) > 0:
        cbar = plt.colorbar(sc, ax=ax7)
        cbar.set_label("Progress")
    ax7.grid(True, alpha=0.3)

    # Panel 8: Energy Efficiency Comparison
    ax8 = plt.subplot(3, 4, 8)
    if baseline_comparison:
        algorithms = list(baseline_comparison.keys())
        energy_values = [
            baseline_comparison[alg]["energy"][-1]
            if isinstance(baseline_comparison[alg]["energy"], list)
            else baseline_comparison[alg]["energy"]
            for alg in algorithms
        ]
        bars = ax8.bar(algorithms, energy_values, alpha=0.7)
        ax8.set_title("Energy Efficiency Comparison", fontsize=12, fontweight="bold")
        ax8.set_ylabel("Energy (kWh)")
        ax8.tick_params(axis="x", rotation=45)
        for bar, value in zip(bars, energy_values):
            height = bar.get_height()
            ax8.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(energy_values) * 0.02,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Panel 9: Utilization Heatmap
    ax9 = plt.subplot(3, 4, 9)
    if "resource_utilization" in eval_results and eval_results["resource_utilization"]:
        util_data = np.array(eval_results["resource_utilization"][:20])
        if len(util_data) > 0:
            im = ax9.imshow(util_data.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax9.set_title("Resource Utilization Profile", fontsize=12, fontweight="bold")
            ax9.set_xlabel("Time Intervals")
            ax9.set_ylabel("Resource ID")
            plt.colorbar(im, ax=ax9, label="Utilization")

    # Panel 10: Response Time Distribution
    ax10 = plt.subplot(3, 4, 10)
    if "response_times" in eval_results and eval_results["response_times"]:
        ax10.hist(eval_results["response_times"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        mean_resp = np.mean(eval_results["response_times"])
        ax10.axvline(mean_resp, color="red", linestyle="--", label=f"Mean: {mean_resp:.2f}s")
        ax10.set_title("Response Time Distribution", fontsize=12, fontweight="bold")
        ax10.set_xlabel("Latency (s)")
        ax10.set_ylabel("Frequency")
        ax10.legend()
        ax10.grid(True, alpha=0.3)

    # Panel 11: KPI Summary
    ax11 = plt.subplot(3, 4, 11)
    metrics = ["Energy\nEfficiency", "Success\nRate", "Resource\nUsage", "Throughput"]
    if "performance_metrics" in eval_results:
        perf = eval_results["performance_metrics"]
        values = [
            perf.get("energy_efficiency", 0),
            perf.get("deadline_met_rate", 0),
            perf.get("avg_cpu_util", 0) / 100.0,
            min(1.0, perf.get("throughput", 0) / 100.0),
        ]
        colors = ["green", "blue", "orange", "red"]
        bars = ax11.bar(metrics, values, color=colors, alpha=0.7)
        ax11.set_title("KPI Summary", fontsize=12, fontweight="bold")
        ax11.set_ylabel("Score")
        ax11.set_ylim([0, 1.15])
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax11.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig("results/dashboard.png", dpi=300)
    plt.savefig("results/system_performance_dashboard.png", dpi=300)
    plt.close()


def plot_baselines(baseline_results: Dict[str, Any]) -> None:
    """Plot multi-scale policy energy and deadline comparisons across workload sizes."""
    if not os.path.exists("results"):
        os.makedirs("results")

    counts = [50, 100, 200, 500, 1000]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for alg, data in baseline_results.items():
        x_vals = counts[: len(data["energy"])]
        axes[0].plot(x_vals, data["energy"], marker="o", label=alg, linewidth=2)
        axes[1].plot(x_vals, data["deadline_met"], marker="s", label=alg, linewidth=2)

    axes[0].set_title("Energy Consumption across Workload Scales", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Tasks")
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Deadline Meeting Rate across Workload Scales", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Tasks")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/baseline_comparison_detailed.png", dpi=300)
    plt.savefig("results/baseline_comparison.png", dpi=300)
    plt.close()
