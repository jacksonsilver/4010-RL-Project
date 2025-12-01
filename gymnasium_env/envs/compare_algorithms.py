import numpy as np
from v0_thin_ice_env import ThinIceEnv
from algorithms import DynaQ, SARSA, QLearning
from scipy import stats

# Tests how good a learned policy is by running it many times
# Counts how often it reaches the goal and calculates average rewards
def test_policy_performance(env, policy, n_tests=50):
    total_reward = 0
    success_count = 0
    total_steps = 0
    successful_steps = []
    
    for i in range(n_tests):
        state = env.reset()[0]  # Start new test run
        done = False
        episode_reward = 0
        steps = 0
        
        # Run until episode ends or too many steps
        while not done and steps < 500:
            # Choose best action from learned policy
            action = np.argmax(policy[state])
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Check if we reached the goal when episode ends
            if done and next_state in env.target:
                success_count += 1
                successful_steps.append(steps)
                # Extra reward for success
                episode_reward += 10  
                break
                
            state = next_state
            
        total_reward += episode_reward
        total_steps += steps
        
    # Calculate averages across all test runs
    avg_reward = total_reward / n_tests
    success_rate = success_count / n_tests
    avg_steps = total_steps / n_tests
    if (successful_steps):
        avg_successful_steps = np.mean(successful_steps) 
    else:
        avg_successful_steps = 0
    # avg_successful_steps = np.mean(successful_steps) if successful_steps else 0
    
    return avg_reward, success_rate, avg_steps, avg_successful_steps

# Trains an algorithm and checks how it improves over time
# Shows learning progress at regular intervals
def track_learning_progress(env, algorithm, algorithm_name, episodes=500, eval_interval=50):
    rewards_over_time = []
    success_rates_over_time = []
    
    # Check performance every few episodes
    for episode in range(0, episodes + 1, eval_interval):
        if episode > 0:
            # Train the algorithm for current number of episodes
            if algorithm_name == "Dyna-Q":
                policy, _ = algorithm(env, max_episode=episode, max_model_step=10)
            else:
                policy, _ = algorithm(env, max_episode=episode)
                
            # Test how good the current policy is
            # avg_reward, success_rate = test_policy_performance(env, policy, n_tests=10)
            avg_reward, success_rate, _, _ = test_policy_performance(env, policy, n_tests=10)
        else:
            avg_reward, success_rate = 0, 0  # No training yet
            
        rewards_over_time.append(avg_reward)
        success_rates_over_time.append(success_rate)
    
    return rewards_over_time, success_rates_over_time

# Compares which algorithm learns faster
# Shows how rewards and success rates improve over training time
def compare_learning_speed(env, algorithms, episodes=500):
    results = {}
    
    for name, algo_func in algorithms.items():
        # Track learning progress for each algorithm
        rewards, success_rates = track_learning_progress(env, algo_func, name, episodes)
        results[name] = {
            'rewards': rewards,
            'success_rates': success_rates,
            'final_performance': (rewards[-1], success_rates[-1])
        }
    
    return results

# Tests different parameters for each algorithm to find the best ones
# Like trying different learning speeds and exploration amounts
# def compare_best_parameters(env, algorithms):
# we will do mutliple trials and more parameter combinations to make sure we have the right algorithm settings
def compare_best_parameters(env, algorithms, n_trials=5):
    parameters_results = {}
    
    # Use same random choices for fair comparison
    # np.random.seed(101225811)
    
    # Different combinations of parameters to test
    parameters_combinations = [
        {'step_size': 0.05, 'epsilon': 0.05},
        {'step_size': 0.05, 'epsilon': 0.1},
        {'step_size': 0.05, 'epsilon': 0.2},
        {'step_size': 0.1, 'epsilon': 0.05},
        {'step_size': 0.1, 'epsilon': 0.1},  
        {'step_size': 0.1, 'epsilon': 0.2},
        {'step_size': 0.2, 'epsilon': 0.05},
        {'step_size': 0.2, 'epsilon': 0.1},
        {'step_size': 0.2, 'epsilon': 0.2},
        # {'step_size': 0.5, 'epsilon': 0.1}, this is weird?
    ]
    
    for algo_name, algo_func in algorithms.items():
        print(f"Testing {algo_name} with different our parameters===")
        algo_results = []
        
        for parameter in parameters_combinations:
            # Run multiple trials for each parameter combo
            trial_rewards = []
            trial_success_rates = []
            trial_steps = []
            
            for trial in range(n_trials):
                # different seed for each trial
                np.random.seed(101225811 + trial)
                
                # Train with current parameters
                if algo_name == "Dyna-Q":
                    policy, _ = algo_func(env, **parameter, max_episode=500, max_model_step=10)
                else:
                    policy, _ = algo_func(env, **parameter, max_episode=500)
                
                # Test performance with these settings
                avg_reward, success_rate, avg_steps, _ = test_policy_performance(env, policy, n_tests=30)
                trial_rewards.append(avg_reward)
                trial_success_rates.append(success_rate)
                trial_steps.append(avg_steps)
            
            # Calculate mean and std across trials
            algo_results.append({
                'params': parameter,
                # 'avg_reward': avg_reward,
                # 'success_rate': success_rate
                'avg_reward': np.mean(trial_rewards),
                'std_reward': np.std(trial_rewards),
                'success_rate': np.mean(trial_success_rates),
                'std_success': np.std(trial_success_rates),
                'avg_steps': np.mean(trial_steps),
                'std_steps': np.std(trial_steps),
                'trial_rewards': trial_rewards,
                'success_rates': trial_success_rates
            })
            
        parameters_results[algo_name] = algo_results
    
    return parameters_results

# Measures how much training each algorithm needs to work well
# Finds which algorithm learns with less practice
def compare_training_needed(env, algorithms, max_episodes=1000, n_trials=3):
    efficiency_results = {}
    target_success = 0.3  # 30% success rate target
    
    for algo_name, algo_func in algorithms.items():
        print(f"Testing training efficiency for {algo_name}...")
        episodes_needed_trials = []
        best_success_rates = []
        
        # Run multiple trials
        for trial in range(n_trials):
            np.random.seed(202530 + trial)
            episodes_needed = None
            best_success_rate = 0
            
            # Try different amounts of training
            for episodes in [100, 200, 300, 500, 1000]:
                if episodes > max_episodes:
                    break
                    
                # train with current num of episode
                if algo_name == "Dyna-Q":
                    policy, _ = algo_func(env, max_episode=episodes, max_model_step=10)
                else:
                    policy, _ = algo_func(env, max_episode=episodes)
                    
                # test performance after the training
                avg_reward, success_rate, _, _ = test_policy_performance(env, policy, n_tests=20)
                
                # record sucesses and best performance
                if success_rate >= target_success and episodes_needed is None:
                    episodes_needed = episodes

                if success_rate > best_success_rate:
                    best_success_rate = success_rate
            
            episodes_needed_trials.append(episodes_needed if episodes_needed else max_episodes)
            best_success_rates.append(best_success_rate)   
        efficiency_results[algo_name] = {
            'avg_episodes_to_target': np.mean(episodes_needed_trials),
            'std_episodes_to_target': np.std(episodes_needed_trials),
            'avg_best_success': np.mean(best_success_rates),
            'std_best_success': np.std(best_success_rates)
        }
    
    return efficiency_results
 
def main():
    # Create the game environment
    env = ThinIceEnv(render_mode=None, level_str='level_6.txt')
    
    # Algorithms to compare
    algorithms = {
        "Q-Learning": QLearning,
        "Dyna-Q": DynaQ,
        "SARSA": SARSA
    }
    
    print("="*80)
    print("Algorithim comparison on Thin Ice Environment for level 6")
    print("="*80)
    
    # testing different hyperparameters for each algorithm
    print("\nTesting different learning rates and epsilons...")
    parameter_results = compare_best_parameters(env, algorithms, n_trials=5)
    
    # show results for each algorithm
    print("\n" + "="*80)
    print("Results - Best Parameters for Each Algorithim")
    print("="*80)
    
    for algo_name, algo_results in parameter_results.items():
        print(f"\n{algo_name}:")
        
        # find best config
        best = max(algo_results, key=lambda x: x['success_rate'])
        
        print(f"  Best parameters: learning rate={best['params']['step_size']}, epsilon={best['params']['epsilon']}")
        print(f"  Success rate: {best['success_rate']:.1%} ± {best['std_success']:.1%}")
        print(f"  Avg reward: {best['avg_reward']:.2f} ± {best['std_reward']:.2f}")
        print(f"  Avg steps: {best['avg_steps']:.1f} ± {best['std_steps']:.1f}")
    
    # check how quickly each algorithm learns
    print("\n" + "="*80)
    print("Training Efficiency")
    print("="*80)
    efficiency_results = compare_training_needed(env, algorithms, max_episodes=1000, n_trials=3)
    
    print("\nHow many episodes to reach 30% success rate:")
    print("-" * 80)
    for algo_name, data in efficiency_results.items():
        avg_episodes = data['avg_episodes_to_target']
        std_episodes = data['std_episodes_to_target']
        avg_success = data['avg_best_success']
        std_success = data['std_best_success']
        print(f"{algo_name}: {avg_episodes:.0f} ± {std_episodes:.0f} episodes (best success: {avg_success:.1%} ± {std_success:.1%})")
    
    print("\n" + "="*80)
    print("Learning Progress Over Time")
    print("="*80)
    learning_results = compare_learning_speed(env, algorithms, episodes=500)
    
    print("\nFinal performance after 500 training episodes:")
    print("-" * 80)
    for algo_name, data in learning_results.items():
        final_reward, final_success = data['final_performance']
        print(f"{algo_name}: success rate = {final_success:.1%}, avg reward = {final_reward:.2f}")
    
    # summary of everything
    print("\n" + "="*80)
    print("Overall Best")
    print("="*80)
    
    # figure out which algorithm + config is best overall
    best_algo = max(parameter_results.keys(), 
                    key=lambda x: max(parameter_results[x], key=lambda r: r['success_rate'])['success_rate'])
    best_config = max(parameter_results[best_algo], key=lambda x: x['success_rate'])
    
    print(f"\nBest algorithm: {best_algo}")
    print(f"Best settings: learning rate = {best_config['params']['step_size']}, epsilon = {best_config['params']['epsilon']}")
    print(f"Success rate: {best_config['success_rate']:.1%} ± {best_config['std_success']:.1%}")
    print(f"Avg reward: {best_config['avg_reward']:.2f} ± {best_config['std_reward']:.2f}")
    print(f"Avg steps: {best_config['avg_steps']:.1f} ± {best_config['std_steps']:.1f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()