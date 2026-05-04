import torch
import numpy as np
from core.environment import CloudEnvironment
from agents.dqn import DQNAgent
from utils.helpers import set_seeds, plot_comprehensive, plot_baselines

def train_agent(env, agent, n_episodes=250):
    history = {k: [] for k in ['rewards', 'losses', 'energies', 'atlp', 'deadline_met', 'throughput', 'moving_avg']}
    best_reward = -float('inf')
    patience = 0
    
    print(f"Starting training on {agent.device}...")
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        episode_deadlines = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Additional reward components from metrics
            # Note: The reward calculation is already inside env.step in the enhanced version
            # but we can add more here if needed.
            
            agent.memory.add(state, action, reward, next_state, done)
            loss = agent.learn()
            
            total_loss += loss
            total_reward += reward
            state = next_state
            steps += 1
            episode_deadlines.append(info['deadline_met'])
            
            if done: break
            
        # Logging and tracking
        avg_loss = total_loss / max(1, steps)
        atlp = np.mean([r.u_j for r in env.resources])
        
        history['rewards'].append(total_reward)
        history['losses'].append(avg_loss)
        history['energies'].append(env.total_energy)
        history['atlp'].append(atlp)
        history['deadline_met'].append(np.mean(episode_deadlines))
        history['throughput'].append(env.metrics_history['throughput'][-1] if env.metrics_history['throughput'] else 0)
        
        # Moving average
        mv_avg = np.mean(history['rewards'][-20:])
        history['moving_avg'].append(mv_avg)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | Energy: {env.total_energy:.2f} | Success: {history['deadline_met'][-1]:.1%}")
            
        # Early stopping & model saving
        if mv_avg > best_reward:
            best_reward = mv_avg
            patience = 0
            torch.save(agent.model.state_dict(), 'best_model.pth')
        else:
            patience += 1
            
        if patience > 50 and ep > 100:
            print("Early stopping triggered.")
            break
            
    return history

def evaluate(env, agent, n_episodes=10):
    print("\nRunning evaluation...")
    agent.model.load_state_dict(torch.load('best_model.pth'))
    
    response_times = []
    util_trace = []
    for _ in range(n_episodes):
        state = env.reset()
        while True:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            response_times.append(info['response_time'])
            state = next_state
            if done: break
        util_trace.append([r.u_j for r in env.resources])
        
    return {
        'response_times': response_times,
        'resource_utilization': util_trace
    }

def run_baseline(env_type, num_tasks):
    env = CloudEnvironment(num_tasks=num_tasks)
    # FCFS
    env.reset()
    while env.current_task_idx < env.num_tasks:
        vm_idx = np.argmin([r.u_j for r in env.resources])
        env.step(vm_idx)
    fcfs_energy = env.total_energy
    fcfs_dl = 1 - np.mean(env.metrics_history['deadline_misses'])
    
    # RR
    env.reset()
    idx = 0
    while env.current_task_idx < env.num_tasks:
        env.step(idx)
        idx = (idx + 1) % env.num_vms
    rr_energy = env.total_energy
    rr_dl = 1 - np.mean(env.metrics_history['deadline_misses'])
    
    return {
        'FCFS': {'energy': fcfs_energy, 'deadline_met': fcfs_dl},
        'RR': {'energy': rr_energy, 'deadline_met': rr_dl}
    }

if __name__ == "__main__":
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = CloudEnvironment(num_tasks=500, num_vms=12)
    agent = DQNAgent(n_input=6, n_actions=12, device=device)
    
    # 1. Training
    history = train_agent(env, agent)
    
    # 2. Evaluation
    eval_results = evaluate(env, agent)
    
    # 3. Baselines
    baseline_results = {
        'FCFS': {'energy': [], 'deadline_met': []},
        'RR': {'energy': [], 'deadline_met': []},
        'DQN': {'energy': [], 'deadline_met': []}
    }
    for size in [50, 100, 200, 500, 1000]:
        b = run_baseline('test', size)
        for alg in ['FCFS', 'RR']:
            baseline_results[alg]['energy'].append(b[alg]['energy'])
            baseline_results[alg]['deadline_met'].append(b[alg]['deadline_met'])
        
        # Eval DQN for this size (simplified)
        env_size = CloudEnvironment(num_tasks=size, num_vms=12)
        state = env_size.reset()
        while True:
            action = agent.select_action(state, eval_mode=True)
            _, _, done, _ = env_size.step(action)
            if done: break
        baseline_results['DQN']['energy'].append(env_size.total_energy)
        baseline_results['DQN']['deadline_met'].append(1 - np.mean(env_size.metrics_history['deadline_misses']))
        
    # 4. Plots
    plot_comprehensive(history, eval_results)
    plot_baselines(baseline_results)
    print("\nAll tasks completed successfully. Plots saved to 'dashboard.png' and 'baselines.png'.")
