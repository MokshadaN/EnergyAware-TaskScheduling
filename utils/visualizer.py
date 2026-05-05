import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict

def comprehensive_plotting(training_history: Dict, eval_results: Dict, baseline_comparison: Dict = None):
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(20, 16))
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(training_history['rewards'], alpha=0.3, label='Iteration Gain', linewidth=0.8)
    ax1.plot(training_history['moving_avg'], 'r-', label='Moving Average', linewidth=2)
    ax1.set_title('Optimization Progress', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Gain')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(training_history['energies'], 'g-', linewidth=1.5)
    ax2.fill_between(range(len(training_history['energies'])), training_history['energies'], alpha=0.3)
    ax2.set_title('Energy Consumption Profile', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Energy (kWh)')
    ax2.grid(True, alpha=0.3)
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(training_history['deadline_met'], 'm-', linewidth=1.5)
    ax3.set_title('Task Success Rate (Deadline Met)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (95%)')
    ax3.legend()
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(training_history['throughput'], 'c-', linewidth=1.5)
    ax4.set_title('System Throughput', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Tasks/Second')
    ax4.grid(True, alpha=0.3)
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(training_history['atlp'], 'orange', linewidth=1.5)
    ax5.fill_between(range(len(training_history['atlp'])), training_history['atlp'], alpha=0.3)
    ax5.set_title('Task Loss Probability (ATLP)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Probability')
    ax5.grid(True, alpha=0.3)
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(training_history['losses'], 'brown', linewidth=1, alpha=0.7)
    ax6.set_title('Optimization Convergence (Loss)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Error Value')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    ax7 = plt.subplot(3, 4, 7)
    energy_subset = training_history['energies'][::max(1, len(training_history['energies'])//50)]
    deadline_subset = training_history['deadline_met'][::max(1, len(training_history['deadline_met'])//50)]
    colors = range(len(energy_subset))
    ax7.scatter(energy_subset, deadline_subset, c=colors, cmap='viridis', alpha=0.6, s=50)
    ax7.set_title('Energy vs Success Trade-off', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Energy (kWh)')
    ax7.set_ylabel('Success Rate')
    if len(energy_subset) > 0:
        cbar = plt.colorbar(ax7.collections[0], ax=ax7)
        cbar.set_label('Progress')
    ax7.grid(True, alpha=0.3)
    ax8 = plt.subplot(3, 4, 8)
    if baseline_comparison:
        algorithms = list(baseline_comparison.keys())
        energy_values = [baseline_comparison[alg]['energy'][-1] if isinstance(baseline_comparison[alg]['energy'], list) 
                        else baseline_comparison[alg]['energy'] for alg in algorithms]
        bars = ax8.bar(algorithms, energy_values, alpha=0.7)
        ax8.set_title('Energy Efficiency Comparison', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Energy (kWh)')
        ax8.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, energy_values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + max(energy_values)*0.02,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    ax9 = plt.subplot(3, 4, 9)
    if 'resource_utilization' in eval_results and eval_results['resource_utilization']:
        util_data = np.array(eval_results['resource_utilization'][:20])
        if len(util_data) > 0:
            im = ax9.imshow(util_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax9.set_title('Resource Utilization Profile', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Time Intervals')
            ax9.set_ylabel('Resource ID')
            plt.colorbar(im, ax=ax9, label='Utilization')
    ax10 = plt.subplot(3, 4, 10)
    if 'response_times' in eval_results and eval_results['response_times']:
        ax10.hist(eval_results['response_times'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax10.axvline(np.mean(eval_results['response_times']), color='red', linestyle='--', 
                    label=f"Mean: {np.mean(eval_results['response_times']):.2f}s")
        ax10.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Latency (s)')
        ax10.set_ylabel('Frequency')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
    ax11 = plt.subplot(3, 4, 11)
    metrics = ['Energy\nEfficiency', 'Success\nRate', 'Resource\nUsage', 'Throughput']
    if 'performance_metrics' in eval_results:
        perf = eval_results['performance_metrics']
        values = [perf.get('energy_efficiency', 0), perf.get('deadline_met', 0), 
                 perf.get('utilization', 0), min(1.0, perf.get('throughput', 0)/100)]
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax11.bar(metrics, values, color=colors, alpha=0.7)
        ax11.set_title('KPI Summary', fontsize=12, fontweight='bold')
        ax11.set_ylabel('Score')
        ax11.set_ylim([0, 1.1])
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/system_performance_dashboard.png', dpi=300)
    plt.close()

def plot_baselines(baseline_results: Dict):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for alg, data in baseline_results.items():
        plt.plot(data['energy'], marker='o', label=alg)
    plt.title('Energy Efficiency Comparison')
    plt.xlabel('Workload Size')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    for alg, data in baseline_results.items():
        plt.plot(data['deadline_met'], marker='s', label=alg)
    plt.title('Success Rate Comparison')
    plt.xlabel('Workload Size')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=300)
    plt.close()
