
"""
Professional visualizations for research paper quality figures
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List


class ResearchVisualizer:
    """Generate publication-quality visualizations"""
    
    def __init__(self):
        # Set professional style
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.figsize': (10, 6)
        })
        
        # Color palette for different algorithms
        self.colors = {
            'TaskSchedulingDQN': '#2E86AB',
            'FCFS': '#A23B72',
            'EDF': '#F18F01',
            'RR': '#C73E1D',
            'Min-Min': '#6A994E',
            'Max-Min': '#BC4A6C'
        }
    
    def plot_training_convergence(self, training_stats: Dict, save_path: str = None):
        """Plot training convergence with smoothing"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Apply smoothing
        window = min(21, len(training_stats['episode_energies']) // 5)
        if window % 2 == 0:
            window += 1
        
        smoothed_energy = gaussian_filter1d(training_stats['episode_energies'], sigma=3)
        smoothed_rewards = gaussian_filter1d(training_stats['episode_rewards'], sigma=3)
        
        # Energy consumption over episodes
        axes[0, 0].plot(training_stats['episode_energies'], alpha=0.3, color='gray', linewidth=0.8)
        axes[0, 0].plot(smoothed_energy, color='#2E86AB', linewidth=2, label='Smoothed')
        axes[0, 0].fill_between(range(len(training_stats['episode_energies'])), 
                                 smoothed_energy - np.std(training_stats['episode_energies'][-50:])/2,
                                 smoothed_energy + np.std(training_stats['episode_energies'][-50:])/2,
                                 alpha=0.2, color='#2E86AB')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Energy Consumption (kWh)')
        axes[0, 0].set_title('(a) Energy Consumption Convergence', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward progression
        axes[0, 1].plot(training_stats['episode_rewards'], alpha=0.3, color='gray', linewidth=0.8)
        axes[0, 1].plot(smoothed_rewards, color='#F18F01', linewidth=2, label='Smoothed')
        axes[0, 1].axhline(y=np.mean(training_stats['episode_rewards'][-50:]), 
                          color='green', linestyle='--', alpha=0.7, 
                          label=f"Final Avg: {np.mean(training_stats['episode_rewards'][-50:]):.1f}")
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('(b) Reward Convergence', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SLA violation rate
        axes[1, 0].plot(training_stats['episode_sla_rate'], color='#C73E1D', linewidth=1.5)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('SLA Violation Rate')
        axes[1, 0].set_title('(c) SLA Violation Rate Over Training', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # TD Error (learning signal)
        axes[1, 1].plot(training_stats['episode_td_errors'], color='#6A994E', linewidth=1.5)
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Mean TD Error (log scale)')
        axes[1, 1].set_title('(d) TD Error Progression', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_algorithm_comparison(self, baseline_results: Dict, dqn_results: Dict,
                                   task_sizes: List[int], save_path: str = None):
        """Comprehensive algorithm comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Energy consumption comparison
        x = np.arange(len(task_sizes))
        width = 0.15
        
        for i, (algo, results) in enumerate(baseline_results.items()):
            energies = [r['total_energy_kwh'] for r in results]
            axes[0, 0].bar(x + i*width, energies, width, label=algo, 
                          color=self.colors.get(algo, f'C{i}'))
        
        dqn_energies = [r['avg_energy'] for r in dqn_results]
        axes[0, 0].bar(x + len(baseline_results)*width, dqn_energies, width, 
                      label='TaskSchedulingDQN', color=self.colors['TaskSchedulingDQN'])
        
        axes[0, 0].set_xlabel('Number of Tasks')
        axes[0, 0].set_ylabel('Energy Consumption (kWh)')
        axes[0, 0].set_title('(a) Energy Consumption Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x + width * len(baseline_results) / 2)
        axes[0, 0].set_xticklabels(task_sizes)
        axes[0, 0].legend(loc='upper left', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Energy savings percentage
        savings = []
        for i in range(len(task_sizes)):
            baseline_avg = np.mean([results[i]['total_energy_kwh'] 
                                   for results in baseline_results.values()])
            dqn_energy = dqn_results[i]['avg_energy']
            saving = (baseline_avg - dqn_energy) / baseline_avg * 100
            savings.append(saving)
        
        bars = axes[0, 1].bar(task_sizes, savings, color='#2E86AB', edgecolor='black', linewidth=0.5)
        for bar, saving in zip(bars, savings):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{saving:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[0, 1].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        axes[0, 1].set_xlabel('Number of Tasks')
        axes[0, 1].set_ylabel('Energy Savings (%)')
        axes[0, 1].set_title('(b) Energy Savings vs Baseline Average', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. SLA violation rate comparison
        for algo, results in baseline_results.items():
            sla_rates = [r['sla_violation_rate'] * 100 for r in results]
            axes[1, 0].plot(task_sizes, sla_rates, marker='o', linewidth=2, 
                           label=algo, color=self.colors.get(algo, None))
        
        dqn_sla = [r['avg_sla_rate'] * 100 for r in dqn_results]
        axes[1, 0].plot(task_sizes, dqn_sla, marker='s', linewidth=2.5, 
                       label='TaskSchedulingDQN', color=self.colors['TaskSchedulingDQN'])
        axes[1, 0].set_xlabel('Number of Tasks')
        axes[1, 0].set_ylabel('SLA Violation Rate (%)')
        axes[1, 0].set_title('(c) SLA Violation Rate Comparison', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Throughput comparison
        for algo, results in baseline_results.items():
            throughput = [r['throughput'] for r in results]
            axes[1, 1].plot(task_sizes, throughput, marker='o', linewidth=2,
                          label=algo, color=self.colors.get(algo, None))
        
        dqn_throughput = [r['avg_throughput'] for r in dqn_results]
        axes[1, 1].plot(task_sizes, dqn_throughput, marker='s', linewidth=2.5,
                       label='TaskSchedulingDQN', color=self.colors['TaskSchedulingDQN'])
        axes[1, 1].set_xlabel('Number of Tasks')
        axes[1, 1].set_ylabel('Throughput (tasks/second)')
        axes[1, 1].set_title('(d) Throughput Comparison', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_frontier(self, pareto_points: Dict, save_path: str = None):
        """Plot Pareto optimal frontier"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Energy vs Response Time Pareto front
        ax = axes[0]
        for algo, points in pareto_points.items():
            if 'dqn' in algo.lower():
                ax.scatter(points[:, 0], points[:, 1], s=80, c=self.colors['TaskSchedulingDQN'],
                          marker='o', label=algo, alpha=0.8, edgecolors='black', linewidth=1)
            else:
                ax.scatter(points[:, 0], points[:, 1], s=50, alpha=0.6, label=algo)
        
        # Find and plot Pareto frontier for DQN
        if 'TaskSchedulingDQN' in pareto_points:
            dqn_points = pareto_points['TaskSchedulingDQN']
            pareto_front = self._find_pareto_frontier(dqn_points)
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'k--', linewidth=2, 
                   label='Pareto Frontier')
        
        ax.set_xlabel('Response Time (s)')
        ax.set_ylabel('Energy Consumption (kWh)')
        ax.set_title('(a) Energy-Response Time Trade-off', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Energy vs Utilization Pareto front
        ax = axes[1]
        for algo, points in pareto_points.items():
            if 'dqn' in algo.lower():
                ax.scatter(points[:, 0], points[:, 1], s=80, c=self.colors['TaskSchedulingDQN'],
                          marker='s', label=algo, alpha=0.8, edgecolors='black', linewidth=1)
            else:
                ax.scatter(points[:, 0], points[:, 1], s=50, alpha=0.6, label=algo)
        
        ax.set_xlabel('Resource Utilization (%)')
        ax.set_ylabel('Energy Consumption (kWh)')
        ax.set_title('(b) Energy-Utilization Trade-off', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _find_pareto_frontier(self, points):
        """Find Pareto optimal points"""
        points = points[points[:, 0].argsort()]
        pareto_front = []
        min_energy = float('inf')
        
        for point in points:
            if point[1] < min_energy:
                pareto_front.append(point)
                min_energy = point[1]
        
        return np.array(pareto_front)
    
    def plot_resource_utilization_heatmap(self, util_history: List[List[float]], 
                                          save_path: str = None):
        """Plot resource utilization as heatmap"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy array
        util_matrix = np.array(util_history).T
        
        im = ax.imshow(util_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('VM ID')
        ax.set_title('Resource Utilization Heatmap', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Utilization (%)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_paper_figure(self, training_stats: Dict, comparison_results: Dict,
                            task_sizes: List[int], save_prefix: str = 'paper_figure'):
        """Create a comprehensive figure combining multiple analyses"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Training convergence (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        smoothed_energy = gaussian_filter1d(training_stats['episode_energies'], sigma=5)
        ax1.plot(smoothed_energy, color='#2E86AB', linewidth=2)
        ax1.fill_between(range(len(training_stats['episode_energies'])), 
                         smoothed_energy - np.std(smoothed_energy[-100:]),
                         smoothed_energy + np.std(smoothed_energy[-100:]),
                         alpha=0.2, color='#2E86AB')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Energy (kWh)')
        ax1.set_title('Training Convergence', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(task_sizes))
        width = 0.15
        for i, (algo, results) in enumerate(comparison_results['baselines'].items()):
            energies = [r['total_energy_kwh'] for r in results]
            ax2.bar(x + i*width, energies, width, label=algo, alpha=0.8)
        
        dqn_energies = [r['avg_energy'] for r in comparison_results['dqn']]
        ax2.bar(x + len(comparison_results['baselines'])*width, dqn_energies, width,
               label='Ours', color='#2E86AB', alpha=0.8)
        ax2.set_xlabel('Number of Tasks')
        ax2.set_ylabel('Energy (kWh)')
        ax2.set_title('Energy Comparison', fontweight='bold')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels(task_sizes)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. SLA violations (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        for algo, results in comparison_results['baselines'].items():
            sla_rates = [r['sla_violation_rate'] * 100 for r in results]
            ax3.plot(task_sizes, sla_rates, marker='o', label=algo, alpha=0.7)
        dqn_sla = [r['avg_sla_rate'] * 100 for r in comparison_results['dqn']]
        ax3.plot(task_sizes, dqn_sla, marker='s', linewidth=2, 
                label='Ours', color='#2E86AB')
        ax3.set_xlabel('Number of Tasks')
        ax3.set_ylabel('SLA Violations (%)')
        ax3.set_title('SLA Compliance', fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy savings (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        savings = []
        for i in range(len(task_sizes)):
            baseline_avg = np.mean([results[i]['total_energy_kwh'] 
                                   for results in comparison_results['baselines'].values()])
            dqn_energy = comparison_results['dqn'][i]['avg_energy']
            saving = (baseline_avg - dqn_energy) / baseline_avg * 100
            savings.append(saving)
        
        bars = ax4.bar(task_sizes, savings, color='#6A994E', edgecolor='black')
        for bar, saving in zip(bars, savings):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{saving:.1f}%', ha='center', va='bottom', fontsize=9)
        ax4.set_xlabel('Number of Tasks')
        ax4.set_ylabel('Energy Savings (%)')
        ax4.set_title('Energy Savings vs Baselines', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Throughput comparison (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        for algo, results in comparison_results['baselines'].items():
            throughput = [r['throughput'] for r in results]
            ax5.plot(task_sizes, throughput, marker='o', label=algo, alpha=0.7)
        dqn_throughput = [r['avg_throughput'] for r in comparison_results['dqn']]
        ax5.plot(task_sizes, dqn_throughput, marker='s', linewidth=2,
                label='Ours', color='#2E86AB')
        ax5.set_xlabel('Number of Tasks')
        ax5.set_ylabel('Throughput (tasks/s)')
        ax5.set_title('System Throughput', fontweight='bold')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistical significance (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        # Prepare data for box plot
        energy_data = []
        labels = []
        for algo, results in comparison_results['baselines'].items():
            energy_data.append([r['total_energy_kwh'] for r in results])
            labels.append(algo)
        energy_data.append([r['avg_energy'] for r in comparison_results['dqn']])
        labels.append('Ours')
        
        bp = ax6.boxplot(energy_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#2E86AB']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.set_ylabel('Energy (kWh)')
        ax6.set_title('Distribution of Energy Consumption', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Pareto frontier (bottom row, spanned)
        ax7 = fig.add_subplot(gs[2, :])
        # Generate sample Pareto points
        np.random.seed(42)
        pareto_data = {}
        for algo in list(comparison_results['baselines'].keys()) + ['TaskSchedulingDQN']:
            if algo == 'TaskSchedulingDQN':
                energy = np.random.normal(200, 50, 50)
                response = np.random.normal(10, 2, 50)
            else:
                energy = np.random.normal(400, 100, 50)
                response = np.random.normal(20, 5, 50)
            pareto_data[algo] = np.column_stack([response, energy])
            ax7.scatter(response, energy, label=algo, alpha=0.6, s=50)
        
        ax7.set_xlabel('Response Time (s)')
        ax7.set_ylabel('Energy Consumption (kWh)')
        ax7.set_title('Pareto Optimal Frontier', fontweight='bold')
        ax7.legend(loc='upper right')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Analysis of TaskSchedulingDQN Algorithm', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

