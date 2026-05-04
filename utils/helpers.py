import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import torch
from typing import Dict

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_comprehensive(history: Dict, eval_results: Dict):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Rewards
    axes[0, 0].plot(history['rewards'], alpha=0.3)
    axes[0, 0].plot(history['moving_avg'], 'r-', linewidth=2)
    axes[0, 0].set_title('Training Rewards')
    
    # Energy
    axes[0, 1].plot(history['energies'], 'g-')
    axes[0, 1].set_title('Energy (kWh)')
    
    # Deadlines
    axes[0, 2].plot(history['deadline_met'], 'm-')
    axes[0, 2].set_title('Deadline Success Rate')
    axes[0, 2].set_ylim([0, 1])
    
    # Throughput
    axes[1, 0].plot(history['throughput'], 'c-')
    axes[1, 0].set_title('Throughput (tasks/s)')
    
    # ATLP
    axes[1, 1].plot(history['atlp'], color='orange')
    axes[1, 1].set_title('ATLP')
    
    # Loss
    axes[1, 2].plot(history['losses'], 'brown')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Training Loss')
    
    # Response Time Dist
    if 'response_times' in eval_results:
        axes[2, 0].hist(eval_results['response_times'], bins=30, color='skyblue')
        axes[2, 0].set_title('Response Time Distribution')
        
    # Utilization Heatmap
    if 'resource_utilization' in eval_results:
        util_data = np.array(eval_results['resource_utilization']).T
        sns.heatmap(util_data, ax=axes[2, 1], cmap='YlOrRd')
        axes[2, 1].set_title('Resource Utilization')
        
    # Summary
    axes[2, 2].axis('off')
    summary = f"Final Stats:\nEnergy: {history['energies'][-1]:.2f}\nDeadline: {history['deadline_met'][-1]:.2%}"
    axes[2, 2].text(0.5, 0.5, summary, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=300)
    plt.show()

def plot_baselines(results: Dict):
    counts = [50, 100, 200, 500, 1000]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for alg, metrics in results.items():
        axes[0].plot(counts, metrics['energy'], marker='o', label=alg)
        axes[1].plot(counts, metrics['deadline_met'], marker='s', label=alg)
        
    axes[0].set_title('Energy Comparison')
    axes[0].set_ylabel('kWh')
    axes[0].legend()
    
    axes[1].set_title('Deadline Met Rate')
    axes[1].set_ylabel('Rate')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('baselines.png')
    plt.show()
