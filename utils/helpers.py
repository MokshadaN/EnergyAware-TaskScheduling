import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_learning_results(history: Dict, baseline_results: Dict = None, eval_metrics: Dict = None):
    """Plot training progress and performance comparisons"""
    # Import locally to avoid circular dependencies if any
    from analysis.advanced import ParetoAnalyzer, RewardAnalyzer
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Rewards
    rewards = history['rewards']
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= 11:
        smoothed = RewardAnalyzer.analyze_reward_trends([(r, r) for r in rewards])['r1_smoothed']
        axes[0, 0].plot(smoothed, 'r-', linewidth=2, label='Smoothed')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Energy
    axes[0, 1].plot(history['energies'], color='green')
    axes[0, 1].set_title('Energy Consumption')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('kWh')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Task Loss Probability
    axes[1, 0].plot(history['atlp'], color='purple')
    axes[1, 0].set_title('Task Loss Probability (ATLP)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('ATLP')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Baselines
    if baseline_results:
        counts = [50, 100, 200, 500, 1000]
        for alg, data in baseline_results.items():
            axes[1, 1].plot(counts[:len(data)], data, marker='o', label=alg)
        axes[1, 1].set_title('Algorithm Comparison')
        axes[1, 1].set_xlabel('Task Count')
        axes[1, 1].set_ylabel('Energy (kWh)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Pareto Frontier
    if eval_metrics and 'all_energies' in eval_metrics and 'all_utils' in eval_metrics:
        e = eval_metrics['all_energies']
        u = eval_metrics['all_utils']
        p = ParetoAnalyzer.analyze_power_vs_utilization(e, u)
        axes[2, 0].scatter(e, u, alpha=0.5, label='Solutions')
        pts = p['pareto_points']
        pts = pts[pts[:, 0].argsort()]
        axes[2, 0].plot(pts[:, 0], pts[:, 1], 'r-o', label='Pareto Front')
        axes[2, 0].set_title('Pareto: Energy vs Utilization')
        axes[2, 0].set_xlabel('Energy (kWh)')
        axes[2, 0].set_ylabel('Utilization (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Summary
    axes[2, 1].text(0.5, 0.5, f'Final Metrics:\nEnergy: {history["energies"][-1]:.2f} kWh\nATLP: {history["atlp"][-1]:.4f}',
                   ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_summary.png', dpi=150)
    plt.show()
