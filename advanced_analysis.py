# advanced_analysis.py
"""
Additional analysis tools for evaluating the TaskSchedulingDQN algorithm
"""

import numpy as np
import torch
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from task_scheduling_dqn import CloudEnvironment, AgentClass, train_agent, evaluate_agent


class ParetoAnalyzer:
    """
    Pareto optimality analysis as described in Figures 6 and 7
    """
    
    @staticmethod
    def find_pareto_frontier(points: np.ndarray, maximize: List[bool] = None) -> np.ndarray:
        """
        Find Pareto optimal points from a set of solutions
        
        Args:
            points: Array of shape (n_solutions, n_objectives)
            maximize: List indicating whether to maximize each objective
        """
        if maximize is None:
            maximize = [False] * points.shape[1]
        
        n_points = points.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j and is_pareto[i]:
                    dominates = True
                    for k in range(points.shape[1]):
                        if maximize[k]:
                            if points[j, k] < points[i, k]:
                                dominates = False
                                break
                        else:
                            if points[j, k] > points[i, k]:
                                dominates = False
                                break
                    if dominates:
                        is_pareto[i] = False
                        break
        
        return is_pareto
    
    @staticmethod
    def analyze_power_vs_utilization(energy_consumption: List[float], 
                                      utilization: List[float]) -> Dict:
        """
        Analyze Pareto frontier for power consumption vs utilization
        (as shown in Figure 6)
        """
        points = np.column_stack([energy_consumption, utilization])
        pareto_indices = ParetoAnalyzer.find_pareto_frontier(
            points, maximize=[False, True]
        )
        
        return {
            'pareto_points': points[pareto_indices],
            'pareto_indices': pareto_indices,
            'optimal_ratio': len(points[pareto_indices]) / len(points)
        }
    
    @staticmethod
    def analyze_response_time_vs_utilization(response_time: List[float],
                                               utilization: List[float]) -> Dict:
        """
        Analyze Pareto frontier for response time vs utilization
        (as shown in Figure 7)
        """
        points = np.column_stack([response_time, utilization])
        pareto_indices = ParetoAnalyzer.find_pareto_frontier(
            points, maximize=[False, True]
        )
        
        return {
            'pareto_points': points[pareto_indices],
            'pareto_indices': pareto_indices,
            'optimal_ratio': len(points[pareto_indices]) / len(points)
        }


class MetricsAnalyzer:
    """
    Performance metrics analyzer as described in Tables II, III, IV
    """
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def compute_performance_metrics(self, task_counts: List[int]) -> Dict:
        """
        Compute comprehensive performance metrics
        Corresponds to Table II in the paper
        """
        results = {
            'task_count': [],
            'energy_consumption': [],
            'cpu_utilization': [],
            'memory_utilization': [],
            'disk_utilization': [],
            'response_times': [],
            'power_consumption': [],
            'total_times': []
        }
        
        for num_tasks in task_counts:
            print(f"Computing metrics for {num_tasks} tasks...")
            
            env = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
            agent_copy = AgentClass(n_input=5, n_actions=10)
            
            # Train briefly
            train_agent(env, agent_copy, num_episodes=30, verbose=False)
            
            # Evaluate
            eval_results = evaluate_agent(env, agent_copy, num_episodes=20)
            
            results['task_count'].append(num_tasks)
            results['energy_consumption'].append(eval_results['avg_energy'])
            results['cpu_utilization'].append(env._get_cpu_utilization())
            results['memory_utilization'].append(env._get_memory_utilization())
            results['disk_utilization'].append(env._get_disk_utilization())
            results['response_times'].append(eval_results['avg_response_time'])
            results['power_consumption'].append(eval_results['avg_energy'] / (max(1, env.current_time) / 3600))
            results['total_times'].append(env.current_time)
        
        return results
    
    def compute_reliability_metrics(self, num_tasks: int) -> Dict:
        """
        Compute reliability metrics (MTBF, FIT)
        Corresponds to Figure 8 in the paper
        """
        env = CloudEnvironment(num_tasks=num_tasks, num_vms=10)
        agent = AgentClass(n_input=5, n_actions=10)
        
        training_history = train_agent(env, agent, num_episodes=50, verbose=False)
        
        # Simulated MTBF and FIT calculations
        # MTBF increases with task count, FIT decreases exponentially
        mtbf = 1000 + 50 * num_tasks  # Simulated
        fit = 1000 * np.exp(-num_tasks / 500)  # Simulated
        
        return {
            'mtbf': mtbf,
            'fit': fit,
            'energy_efficiency': np.mean(training_history['energies'][-10:]) / (num_tasks + 1)
        }


class RewardAnalyzer:
    """
    Analyzer for reward values R1 and R2
    Corresponds to Figure 5 in the paper
    """
    
    @staticmethod
    def analyze_reward_trends(reward_history: List[Tuple[float, float]], 
                              window_length: int = 11,
                              polyorder: int = 2) -> Dict:
        """
        Apply Savitzky-Golay filter to smooth reward trends
        
        As described in the paper: "trend is highlighted using the 
        Savitzky-Golay filter for the purpose of pointing out the 
        increasing trend in the normalized reward values"
        """
        rewards_r1 = [r[0] for r in reward_history]
        rewards_r2 = [r[1] for r in reward_history]
        
        # Apply Savitzky-Golay filter
        if len(rewards_r1) >= window_length:
            smoothed_r1 = savgol_filter(rewards_r1, window_length, polyorder)
            smoothed_r2 = savgol_filter(rewards_r2, window_length, polyorder)
        else:
            smoothed_r1 = rewards_r1
            smoothed_r2 = rewards_r2
        
        return {
            'r1_raw': rewards_r1,
            'r2_raw': rewards_r2,
            'r1_smoothed': smoothed_r1,
            'r2_smoothed': smoothed_r2,
            'cumulative_reward': np.cumsum([r1 + r2 for r1, r2 in reward_history])
        }


def plot_comprehensive_metrics(metrics_results: Dict):
    """
    Generate comprehensive visualizations as shown in Figure 8
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Metric 1: Resource Efficiency
    axes[0, 0].bar(metrics_results['task_count'], metrics_results['cpu_utilization'])
    axes[0, 0].set_title('Resource Efficiency')
    axes[0, 0].set_xlabel('Number of Tasks')
    axes[0, 0].set_ylabel('CPU Utilization (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metric 2: Energy Trends
    axes[0, 1].plot(metrics_results['task_count'], metrics_results['energy_consumption'], 
                    marker='o', linewidth=2)
    axes[0, 1].set_title('Energy Trends')
    axes[0, 1].set_xlabel('Number of Tasks')
    axes[0, 1].set_ylabel('Energy Consumption (kWh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metric 3: Reliability Metrics
    axes[0, 2].plot(metrics_results['task_count'], metrics_results.get('mtbf', [1000]*len(metrics_results['task_count'])), 
                    marker='s', label='MTBF')
    axes[0, 2].set_title('Reliability Metrics')
    axes[0, 2].set_xlabel('Number of Tasks')
    axes[0, 2].set_ylabel('MTBF (s)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Metric 4: Performance Scaling
    axes[1, 0].plot(metrics_results['task_count'], metrics_results['response_times'], 
                    marker='^', color='green', linewidth=2)
    axes[1, 0].set_title('Performance Scaling')
    axes[1, 0].set_xlabel('Number of Tasks')
    axes[1, 0].set_ylabel('Response Time (s)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metric 5: Load Distribution
    axes[1, 1].bar(metrics_results['task_count'], metrics_results['memory_utilization'], 
                   alpha=0.7, label='Memory')
    axes[1, 1].bar(metrics_results['task_count'], metrics_results['disk_utilization'], 
                   alpha=0.7, label='Disk')
    axes[1, 1].set_title('Load Distribution')
    axes[1, 1].set_xlabel('Number of Tasks')
    axes[1, 1].set_ylabel('Utilization (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Metric 6: Overall Efficiency
    efficiency = [e / t for e, t in zip(metrics_results['energy_consumption'], 
                                         metrics_results['task_count'])]
    axes[1, 2].plot(metrics_results['task_count'], efficiency, 
                    marker='D', color='red', linewidth=2)
    axes[1, 2].set_title('Energy Efficiency per Task')
    axes[1, 2].set_xlabel('Number of Tasks')
    axes[1, 2].set_ylabel('Energy per Task (kWh)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()