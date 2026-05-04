# hyperparameter_tuning.py (CORRECTED)
"""
Hyperparameter optimization for TaskSchedulingDQN
"""

from typing import Dict, Any, List
import itertools
import numpy as np

# Import from main implementation
from task_scheduling_dqn import (
    CloudEnvironment, AgentClass, train_agent, evaluate_agent
)


class HyperparameterTuner:
    """
    Systematic hyperparameter tuning for the DQN agent
    """
    
    def __init__(self, env_creator, agent_creator):
        self.env_creator = env_creator
        self.agent_creator = agent_creator
    
    def grid_search(self, param_grid: Dict[str, List[Any]], 
                    num_episodes: int = 50,
                    eval_episodes: int = 5) -> Dict:
        """
        Perform grid search over hyperparameters
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_params = None
        best_score = float('inf')
        results = []
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            print(f"\nTesting parameters: {params}")
            
            # Run with these parameters
            scores = []
            for seed in range(3):  # 3 runs for stability
                np.random.seed(seed)
                env = self.env_creator()
                agent = self.agent_creator(**params)
                
                # Train
                train_agent(env, agent, num_episodes=num_episodes, verbose=False)
                
                # Evaluate
                eval_results = evaluate_agent(env, agent, num_episodes=eval_episodes)
                scores.append(eval_results['avg_energy'])
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            results.append({
                'params': params,
                'avg_energy': avg_score,
                'std_energy': std_score
            })
            
            print(f"  Average Energy: {avg_score:.2f} ± {std_score:.2f} kWh")
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"  *** NEW BEST! ***")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    @staticmethod
    def get_default_param_grid() -> Dict:
        """
        Returns default hyperparameter grid based on paper recommendations
        """
        return {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'gamma': [0.95, 0.99, 0.995],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'batch_size': [16, 32, 64],
            'target_update_freq': [50, 100, 200],
            'soft_update_param': [0.001, 0.01, 0.05]
        }


def run_tuning():
    """Run hyperparameter tuning"""
    print("=" * 60)
    print("Hyperparameter Tuning for TaskSchedulingDQN")
    print("=" * 60)
    
    def create_env():
        return CloudEnvironment(num_tasks=200, num_vms=10)
    
    def create_agent(learning_rate=0.001, gamma=0.99, epsilon_decay=0.995,
                     batch_size=32, target_update_freq=100, soft_update_param=0.01):
        return AgentClass(
            n_input=5,
            n_actions=10,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            soft_update_param=soft_update_param
        )
    
    tuner = HyperparameterTuner(create_env, create_agent)
    
    # Use smaller grid for testing
    param_grid = {
        'learning_rate': [0.0005, 0.001],
        'gamma': [0.99],
        'epsilon_decay': [0.995],
        'batch_size': [32],
        'target_update_freq': [100],
        'soft_update_param': [0.01]
    }
    
    results = tuner.grid_search(param_grid, num_episodes=30, eval_episodes=20)
    
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Energy Score: {results['best_score']:.2f} kWh")


if __name__ == "__main__":
    run_tuning()