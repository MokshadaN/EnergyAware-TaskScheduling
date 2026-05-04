import numpy as np
from improved_task_scheduling import (
    RealisticCloudEnvironment, EnhancedAgent, train_with_metrics,
    evaluate_agent_episode, run_baseline_algorithms
)
from enhanced_visualizations import ResearchVisualizer
import time
import pickle
from datetime import datetime


def run_experiments():
    """Run all experiments and generate comprehensive results"""
    
    print("=" * 70)
    print("ENERGY-AWARE TASK SCHEDULING USING DEEP Q-LEARNING")
    print("Comprehensive Experimental Analysis")
    print("=" * 70)
    
    # Configuration
    task_sizes = [50, 100, 200, 500, 1000]
    num_vms = 10
    training_episodes = 150
    
    visualizer = ResearchVisualizer()
    
    # Store results
    all_results = {
        'dqn_eval_results': [],
        'baseline_results': {},
        'training_stats': None
    }
    
    # 1. Train DQN agent on largest task set
    print(f"\n[1] Training DQN Agent on {task_sizes[-1]} tasks...")
    print("-" * 50)
    
    env = RealisticCloudEnvironment(num_tasks=task_sizes[-1], num_vms=num_vms)
    agent = EnhancedAgent(n_input=5, n_actions=num_vms)
    
    start_time = time.time()
    training_stats = train_with_metrics(env, agent, num_episodes=training_episodes, eval_freq=25)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    all_results['training_stats'] = training_stats
    
    # Plot training convergence
    visualizer.plot_training_convergence(training_stats, save_path='training_convergence.png')
    
    # 2. Evaluate on different task sizes
    print("\n[2] Evaluating DQN Agent on Different Task Loads...")
    print("-" * 50)
    
    dqn_results = []
    for num_tasks in task_sizes:
        print(f"\nEvaluating on {num_tasks} tasks...")
        env_eval = RealisticCloudEnvironment(num_tasks=num_tasks, num_vms=num_vms)
        eval_results = evaluate_agent_episode(env_eval, agent, num_episodes=10)
        dqn_results.append(eval_results)
        
        print(f"  Energy: {eval_results['avg_energy']:.2f} ± {eval_results['std_energy']:.2f} kWh")
        print(f"  SLA Rate: {eval_results['avg_sla_rate']:.2%}")
        print(f"  Throughput: {eval_results['avg_throughput']:.2f} tasks/s")
    
    all_results['dqn_eval_results'] = dqn_results
    
    # 3. Run baseline algorithms
    print("\n[3] Running Baseline Algorithms...")
    print("-" * 50)
    
    for num_tasks in task_sizes:
        print(f"\nFor {num_tasks} tasks:")
        env_baseline = RealisticCloudEnvironment(num_tasks=num_tasks, num_vms=num_vms)
        
        if num_tasks not in all_results['baseline_results']:
            all_results['baseline_results'][num_tasks] = {}
        
        # Run baselines
        baseline_metrics = run_baseline_algorithms(env_baseline, num_tasks)
        
        for algo, metrics in baseline_metrics.items():
            all_results['baseline_results'][num_tasks][algo] = metrics
            print(f"  {algo}: {metrics['total_energy_kwh']:.2f} kWh, "
                  f"SLA: {metrics['sla_violation_rate']:.2%}")
    
    # 4. Prepare comparison data for visualization
    comparison_results = {
        'baselines': {},
        'dqn': dqn_results
    }
    
    # Organize baseline results by algorithm
    algorithms = list(all_results['baseline_results'][task_sizes[0]].keys())
    for algo in algorithms:
        comparison_results['baselines'][algo] = [
            all_results['baseline_results'][n][algo] for n in task_sizes
        ]
    
    # 5. Generate comprehensive visualizations
    print("\n[4] Generating Visualizations...")
    print("-" * 50)
    
    # Main comparison figure
    visualizer.plot_algorithm_comparison(
        comparison_results['baselines'],
        dqn_results,
        task_sizes,
        save_path='algorithm_comparison.png'
    )
    
    # Resource utilization heatmap
    visualizer.plot_resource_utilization_heatmap(
        env.resource_util_history[-200:],  # Last 200 steps
        save_path='resource_heatmap.png'
    )
    
    # Comprehensive paper figure
    visualizer.create_paper_figure(
        training_stats,
        comparison_results,
        task_sizes,
        save_prefix='paper_figure'
    )
    
    # 5. Generate summary table
    print("\n[5] Generating Results Summary...")
    print("=" * 70)
    print("\nTABLE: Performance Comparison Across Task Loads")
    print("-" * 70)
    print(f"{'Tasks':>8} | {'Algorithm':<18} | {'Energy (kWh)':<12} | {'SLA Rate':<10} | {'Throughput':<12}")
    print("-" * 70)
    
    for i, num_tasks in enumerate(task_sizes):
        # Baselines average
        baseline_avg = np.mean([comp[num_tasks][algo]['total_energy_kwh'] 
                               for algo in algorithms])
        baseline_sla = np.mean([comp[num_tasks][algo]['sla_violation_rate'] 
                               for algo in algorithms])
        
        print(f"{num_tasks:>8} | {'Baseline Avg':<18} | {baseline_avg:<12.2f} | {baseline_sla:<10.2%} | {'-':<12}")
        print(f"{'':8} | {'TaskSchedulingDQN':<18} | {dqn_results[i]['avg_energy']:<12.2f} | "
              f"{dqn_results[i]['avg_sla_rate']:<10.2%} | {dqn_results[i]['avg_throughput']:<12.2f}")
        if i < len(task_sizes) - 1:
            print("-" * 70)
    
    print("=" * 70)
    
    # Calculate improvement percentages
    print("\nSUMMARY OF IMPROVEMENTS:")
    print("-" * 50)
    
    for i, num_tasks in enumerate(task_sizes):
        baseline_avg = np.mean([comp[num_tasks][algo]['total_energy_kwh'] 
                               for algo in algorithms])
        dqn_energy = dqn_results[i]['avg_energy']
        improvement = (baseline_avg - dqn_energy) / baseline_avg * 100
        
        dqn_sla = dqn_results[i]['avg_sla_rate']
        baseline_sla = np.mean([comp[num_tasks][algo]['sla_violation_rate'] 
                               for algo in algorithms])
        sla_reduction = (baseline_sla - dqn_sla) / baseline_sla * 100 if baseline_sla > 0 else 0
        
        print(f"\n{num_tasks} Tasks:")
        print(f"  Energy Improvement: {improvement:.1f}%")
        print(f"  SLA Reduction: {sla_reduction:.1f}%")
        print(f"  Throughput: {dqn_results[i]['avg_throughput']:.2f} tasks/s")
    
    # Save results
    with open('experiment_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n[✓] All experiments completed successfully!")
    print("Results saved to:")
    print("  - experiment_results.pkl")
    print("  - training_convergence.png")
    print("  - algorithm_comparison.png")
    print("  - resource_heatmap.png")
    print("  - paper_figure_comprehensive.png")
    
    return all_results


if __name__ == "__main__":
    results = run_experiments()