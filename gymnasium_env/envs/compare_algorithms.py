import numpy as np
from v0_thin_ice_env import ThinIceEnv
from algorithms import DynaQ, SARSA

# Tests how good a learned policy is by running it many times
# Counts how often it reaches the goal and calculates average rewards
def test_policy_performance(env, policy, n_tests=50):
    total_reward = 0
    success_count = 0
    
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
                # Extra reward for success
                episode_reward += 10  
                break
                
            state = next_state
            
        total_reward += episode_reward
        
    # Calculate averages across all test runs
    avg_reward = total_reward / n_tests
    success_rate = success_count / n_tests
    
    return avg_reward, success_rate

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
            avg_reward, success_rate = test_policy_performance(env, policy, n_tests=10)
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
def compare_best_parameters(env, algorithms):
    parameters_results = {}
    
    # Use same random choices for fair comparison
    np.random.seed(101225811)
    
    # Different combinations of settings to test
    parameters_combinations = [
        {'step_size': 0.05, 'epsilon': 0.1},
        {'step_size': 0.1, 'epsilon': 0.1},  
        {'step_size': 0.5, 'epsilon': 0.1}, 
        {'step_size': 0.1, 'epsilon': 0.05},
        {'step_size': 0.1, 'epsilon': 0.5}, 
    ]
    
    for algo_name, algo_func in algorithms.items():
        algo_results = []
        for parameter in parameters_combinations:
            # Train with different algo algo and parameter
            if algo_name == "Dyna-Q":
                policy, _ = algo_func(env, **parameter, max_episode=300, max_model_step=10)
            else:
                policy, _ = algo_func(env, **parameter, max_episode=300)
            
            # Test performance with these settings
            avg_reward, success_rate = test_policy_performance(env, policy)
            algo_results.append({
                'params': parameter,
                'avg_reward': avg_reward,
                'success_rate': success_rate
            })
        parameters_results[algo_name] = algo_results
    
    return parameters_results

# Measures how much training each algorithm needs to work well
# Finds which algorithm learns with less practice
def compare_training_needed(env, algorithms, max_episodes=1000):
    efficiency_results = {}
    target_success = 0.1  # 10% success rate target
    
    for algo_name, algo_func in algorithms.items():
        episodes_needed = None
        best_success_rate = 0
        
        # Try different amounts of training
        for episodes in [100, 200, 300, 500, 1000]:
            if episodes > max_episodes:
                break
                
            # Train with current episode count
            if algo_name == "Dyna-Q":
                policy, _ = algo_func(env, max_episode=episodes, max_model_step=10)
            else:
                policy, _ = algo_func(env, max_episode=episodes)
                
            # Test performance after this much training
            avg_reward, success_rate = test_policy_performance(env, policy, n_tests=20)
            
            # Record when target is first reached
            if success_rate >= target_success and episodes_needed is None:
                episodes_needed = episodes
                
            # Track best performance seen
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                
        efficiency_results[algo_name] = {
            'episodes_to_target': episodes_needed,
            'best_success_rate': best_success_rate
        }
    
    return efficiency_results
 
def main():
    # Create the game environment
    env = ThinIceEnv(render_mode=None, level_str='level_5.txt')
    
    # Algorithms to compare
    algorithms = {
        "Dyna-Q": DynaQ,
        "SARSA": SARSA
    }
    
    # Compare how fast they learn
    learning_results = compare_learning_speed(env, algorithms, episodes=500)
    
    # Compare which parameters work best
    parameter_results = compare_best_parameters(env, algorithms)
    
    # Compare how much training they need
    efficiency_results = compare_training_needed(env, algorithms, max_episodes=500)
    
    # Show all results
    # print_comparison_summary(learning_results, settings_results, efficiency_results)
    print("How well each algorithm performs after full training:")
    for algo_name, data in learning_results.items():
        final_reward, final_success = data['final_performance']
        print(f"{algo_name}:")
        print(f"  Final Average Reward: {final_reward:.3f}")
        print(f"  Final Success Rate: {final_success:.2f}")
    
    print("Best paramter configuration found for each algorithm:")
    for algo_name, algo_results in parameter_results.items():
        print(f"{algo_name}:")
        best_result = max(algo_results, key=lambda x: x['avg_reward'])
        print(f"  Best parameters: {best_result['params']}")
        print(f"  Best reward: {best_result['avg_reward']:.3f}")
        print(f"  Best success rate: {best_result['success_rate']:.2f}")
    
    print("How quickly each algorithm reaches good performance:")
    for algo_name, data in efficiency_results.items():
        episodes_needed = data['episodes_to_target']
        best_success = data['best_success_rate']
        if episodes_needed:
            print(f"{algo_name}: Reached target in {episodes_needed} episodes")
        else:
            print(f"{algo_name}: Best success rate = {best_success:.2f} (did not reach target)")

if __name__ == "__main__":
    main()