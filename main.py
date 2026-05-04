import torch
import numpy as np
from core.environment import CloudEnvironment, calculate_atlp
from agents.dqn import DQNAgent
from utils.helpers import set_seeds, plot_learning_results
from analysis.advanced import MetricsAnalyzer

def run_experiment(n_episodes=100, n_tasks=500, n_vms=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    env = CloudEnvironment(num_tasks=n_tasks, num_vms=n_vms)
    agent = DQNAgent(n_input=5, n_actions=n_vms, device=device)
    
    history = {
        'rewards': [],
        'losses': [],
        'energies': [],
        'atlp': []
    }
    
    print(f"Starting training: {n_episodes} episodes...")
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Metrics for reward processing
            m = {
                'uptime': state[0], 'cpu': state[1], 'mem': state[2],
                'disk': state[3], 'ram': state[4]
            }
            dq1, r1, dq2, r2 = agent.process_rewards(state, action, next_state, m)
            
            # Final composite reward
            final_reward = reward + 0.1 * (r1 + r2)
            agent.memory.add(state, action, final_reward, next_state, done)
            
            loss = agent.train_step()
            total_loss += loss
            total_reward += final_reward
            state = next_state
            steps += 1
            
            if done: break
            
        if (ep + 1) % 10 == 0:
            agent.update_target()
            
        atlp = calculate_atlp(env)
        history['rewards'].append(total_reward)
        history['losses'].append(total_loss / steps)
        history['energies'].append(env.total_energy)
        history['atlp'].append(atlp)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | Energy: {env.total_energy:.2f} | ATLP: {atlp:.4f}")
            
    return env, agent, history

def evaluate(env, agent, n_episodes=10):
    print("\nRunning evaluation...")
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    
    results = []
    for _ in range(n_episodes):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done: break
        results.append({
            'energy': env.total_energy,
            'util': env._cpu_util() / 100.0
        })
        
    agent.epsilon = orig_eps
    return {
        'avg_energy': np.mean([r['energy'] for r in results]),
        'all_energies': [r['energy'] for r in results],
        'all_utils': [r['util'] for r in results]
    }

if __name__ == "__main__":
    set_seeds(42)
    
    # 1. Train
    env, agent, history = run_experiment(n_episodes=100)
    
    # 2. Evaluate
    eval_metrics = evaluate(env, agent)
    
    # 3. Analyze reliability
    rel = MetricsAnalyzer(env, agent).get_reliability_stats(500)
    print(f"\nReliability: MTBF={rel['mtbf']:.2f}s, FIT={rel['fit']:.2f}")
    
    # 4. Plot
    plot_learning_results(history, eval_metrics=eval_metrics)
    print("\nExperiment complete. Results saved.")
