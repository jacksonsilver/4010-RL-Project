import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import v0_thin_ice_env as ti #including it so it registers 
import os

from thin_ice_training_agent import ThinIceTrainingAgent
from gymnasium_env.envs.components.decaying_epsilon import DecayingEpsilon

class ThinIceQLearningAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='Level0.txt'):
        super().__init__("QLearning", env_id, level_str)

    def train(self, gamma: float = 0.9, step_size: float = 0.1, epsilon: float = 0.1, end_epsilon: float = 0.05, decay_rate: int = 4000000, n_episodes: int = 1000):
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)

        # Initialize Q table with random values for n_states x n_actions
        q = np.random.rand(env.unwrapped.n_states,
            env.unwrapped.n_actions)
        # Ensure that goal (terminal) state(s) is initialized to 0
        targets = env.unwrapped.target
        q[targets, :] = 0

        #keeping count of number of steps per episode (is it becoming more efficient or not)
        number_of_steps = np.zeros(n_episodes)
        rewards_per_episode = np.zeros(n_episodes)
        epsilon_history = []
        successful_episodes = 0

        decaying_epsilon = DecayingEpsilon(start_epsilon=epsilon, end_epsilon=end_epsilon, decay_rate=decay_rate)

        for i in range(n_episodes):
            print(f'Episode: {i}')

            #Reset env before each episode
            state = env.reset()[0]
            step_count = 0
            episode_reward = 0
            terminated = False

            while (not terminated):
                step_count += 1

                # From the state value, get the available actions mask, which is an int where each bit represents an action
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                # If there are no available actions, choose randomly from all actions
                if len(available_actions) == 0:
                    available_actions = env.unwrapped.get_termination_actions(state)

                # Choose action from state based on epsilon-greedy policy
                if np.random.rand() < decaying_epsilon.get_epsilon():
                    action = np.random.choice(available_actions)
                else:
                    # Get the action with the highest Q-value that's also in available actions
                    action = available_actions[np.argmax(q[state, available_actions])]

                next_state,reward,terminated,_,_ = env.step(action)
                episode_reward += reward

                # Update q table with formula from class
                next_action = np.argmax(q[next_state])
                q[state, action] = q[state, action] + step_size * (
                        reward + gamma * np.max(q[next_state, next_action]) - q[state, action]
                )
                
                # Update State
                state = next_state
            
            number_of_steps[i] = step_count
            rewards_per_episode[i] = episode_reward
            epsilon_history.append(decaying_epsilon.get_epsilon())
            
            # Check if episode was successful (reached target)
            if state in targets:
                successful_episodes += 1

            decaying_epsilon.update(i)

        #for loop done
        env.close()

        self.generate_graph(number_of_steps)

        # Create new graph for rewards and epsilon decay
        plt.figure(figsize=(12, 5))

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(n_episodes)
        for x in range(n_episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel("Episode Number")
        plt.ylabel("Reward per Episode (rolling sum of last 100)")
        plt.title("Rewards Over Episodes")
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        plt.xlabel("Episode Number")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Episodes")
        
        # Save plots
        plt.tight_layout()
        graph_path = os.path.join(self.getGraphFolderPath(self.algorithm_name), self.reference_name + '_rewards_epsilon.png')
        plt.savefig(graph_path)
        print(f"Saved rewards/epsilon graph to: {graph_path}")

        f = open(os.path.join(self.getPkFolderPath(self.algorithm_name), self.reference_name + '_solution.pk1'), "wb")
        pickle.dump(q,f)
        f.close()

        return q, rewards_per_episode, successful_episodes/n_episodes

    def run_policy(self, q, env, n_episodes: int = 100, max_steps: int = 500):
        """Evaluate a trained Q-table policy and return success rate and visited tile percentages."""
        visited_tile_percents = []
        successful_episodes = 0
        targets = env.unwrapped.target
        
        for _ in range(n_episodes):
            state = env.reset()[0]
            terminated = False
            step_count = 0
            
            # Get total visitable tiles AFTER reset (when level is fresh)
            total_visitable_tiles = env.unwrapped.level.get_visitable_tile_count()

            while not terminated and step_count < max_steps:
                step_count += 1
                
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)
                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)
                
                if len(available_actions) == 0:
                    available_actions = env.unwrapped.get_termination_actions(state)
                
                # Greedy action selection (no exploration)
                action = available_actions[np.argmax(q[state, available_actions])]
                
                next_state, reward, terminated, _, info = env.step(action)
                state = next_state
            
            # Check if episode was successful (reached target)
            if state in targets:
                successful_episodes += 1
            
            # Calculate percentage of tiles visited
            try:
                visited_count = env.unwrapped.level.get_visited_tile_count()
                # Cap at 100% in case of any edge cases
                visited_percent = min((visited_count / total_visitable_tiles) * 100, 100.0)
            except:
                visited_percent = 0.0
            visited_tile_percents.append(visited_percent)
        
        # Return success rate (0-1) and visited tile percentages
        success_rate = successful_episodes / n_episodes
        return visited_tile_percents, success_rate

    def _create_results_table(self, title, row_labels, col_labels, data, filename):
        """Helper function to create and save a results table as an image."""
        # Calculate figure size based on number of rows and columns
        fig_width = max(10, len(col_labels) * 2.5 + 3)
        fig_height = max(4, len(row_labels) * 0.4 + 2)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Format data to 4 decimal places
        formatted_data = [[f"{val:.4f}" for val in row] for row in data]
        
        table = ax.table(
            cellText=formatted_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        
        # Style header cells
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Style row label cells with alternating colors for better readability
        for i in range(len(row_labels)):
            if i % 3 == 0:  # Group by step_size (every 3 rows is a new episode count)
                table[(i + 1, -1)].set_facecolor('#D9E2F3')
            else:
                table[(i + 1, -1)].set_facecolor('#E9EFF7')
            table[(i + 1, -1)].set_text_props(fontweight='bold', fontsize=8)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = os.path.join(self.getGraphFolderPath(self.algorithm_name), filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved table to: {save_path}")

    def runQLearningTesting(self, n_runs: int = 5):
        def repeatExperiments(self, env, n_episodes=1000, step_size=0.1, epsilon=0.1, end_epsilon=0.05, decay_rate=4000000, gamma=0.9):
            total_visit_percents = []
            avg_rewards_per_episode = []
            avg_successful_episodes = []  # Now tracks evaluation success rate (0-1)
            
            for i in range(n_runs):
                print(f"TRAINING Run {i+1}/{n_runs}")
                q, rewards_per_episode, training_successes = self.train(
                    n_episodes=n_episodes, 
                    epsilon=epsilon, 
                    step_size=step_size, 
                    end_epsilon=end_epsilon, 
                    decay_rate=decay_rate, 
                    gamma=gamma
                )

                print("EVALUATE")
                visited_tile_percents, eval_success_rate = self.run_policy(q=q, env=env)
                total_visit_percents.extend(visited_tile_percents)
                avg_rewards_per_episode.append(np.mean(rewards_per_episode))
                avg_successful_episodes.append(training_successes) 
            
            return np.mean(total_visit_percents), np.mean(avg_rewards_per_episode), np.mean(avg_successful_episodes) * 100

        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str, render_mode=None)

        # # Go through different episode counts and step sizes to see performance
        # step_sizes = [0.1, 0.01, 0.001]
        # episode_counts = [2000, 10000, 20000]
        # init_epsilons = [0.1, 0.3, 0.5]

        # results = []
        # i = 0

        # # # Test different combinations of step_size, episodes, and epsilon
        # for episode in episode_counts:
        #     for step_size in step_sizes:
        #         for epsilon in init_epsilons:
        #             percent_correct, avg_reward, avg_successful_episodes = repeatExperiments(
        #                 self, env, n_episodes=episode, step_size=step_size, epsilon=epsilon
        #             )
        #             results.append((percent_correct, avg_reward, avg_successful_episodes))

        # # Create 3 consolidated tables showing ALL combinations
        # # Rows: step_size × episode_count combinations
        # # Columns: init_epsilon values
        # percent_data = []
        # reward_data = []
        # success_data = []
        # row_labels = []
        
        # idx = 0
        # for episode in episode_counts:
        #     for step_size in step_sizes:
        #         percent_row = []
        #         reward_row = []
        #         success_row = []
        #         for epsilon in init_epsilons:
        #             percent_row.append(results[idx][0])
        #             reward_row.append(results[idx][1])
        #             success_row.append(results[idx][2])
        #             idx += 1
        #         percent_data.append(percent_row)
        #         reward_data.append(reward_row)
        #         success_data.append(success_row)
        #         row_labels.append(f"α={step_size}, ep={episode}")
        
        # col_labels = [f"ε={e}" for e in init_epsilons]
        
        # self._create_results_table("% Tiles Visited (All Hyperparameter Combinations)", row_labels, col_labels, percent_data, "hyperparams_percent_visited.png")
        # self._create_results_table("Avg Reward per Episode (All Hyperparameter Combinations)", row_labels, col_labels, reward_data, "hyperparams_avg_reward.png")
        # self._create_results_table("Avg Success Rate (All Hyperparameter Combinations)", row_labels, col_labels, success_data, "hyperparams_success_rate.png")

        # for episode in episode_counts:    
        #     for step_size in step_sizes:
        #         for epsilon in init_epsilons:
        #             percent_correct = results[i][0]
        #             avg_reward = results[i][1]
        #             avg_successful_episodes = results[i][2]
        #             i += 1
        #             print(f"alpha, episode, epsilon: {step_size}, {episode}, {epsilon}, Avg. Percent Correct: {percent_correct}, avg reward: {avg_reward}, avg successful episodes: {avg_successful_episodes}")

        # # Test different epsilon decay configurations
        init_epsilons = [0.1, 0.3, 0.5]
        end_epsilons = [0.001, 0.05, 0.1]
        decay_rates = [1000000, 4000000, 8000000]

        results_decay = []
        j = 0

        for init_epsilon in init_epsilons:
            for end_epsilon in end_epsilons:
                for rate in decay_rates:
                    percent_correct, avg_reward, avg_successful_episodes = repeatExperiments(
                        self, env, n_episodes=20000, step_size=0.1, epsilon=init_epsilon, 
                        end_epsilon=end_epsilon, decay_rate=rate
                    )
                    results_decay.append((percent_correct, avg_reward, avg_successful_episodes))
        
        # Create 3 consolidated tables for epsilon decay testing
        # Rows: init_epsilon × end_epsilon combinations
        # Columns: decay_rate values
        percent_data_decay = []
        reward_data_decay = []
        success_data_decay = []
        row_labels_decay = []
        
        idx = 0
        for init_epsilon in init_epsilons:
            for end_epsilon in end_epsilons:
                percent_row = []
                reward_row = []
                success_row = []
                for rate in decay_rates:
                    percent_row.append(results_decay[idx][0])
                    reward_row.append(results_decay[idx][1])
                    success_row.append(results_decay[idx][2])
                    idx += 1
                percent_data_decay.append(percent_row)
                reward_data_decay.append(reward_row)
                success_data_decay.append(success_row)
                row_labels_decay.append(f"init_ε={init_epsilon}, end_ε={end_epsilon}")
        
        # Format decay rates for column labels (e.g., 1M, 4M, 8M)
        col_labels_decay = [f"decay={r//1000000}M" for r in decay_rates]
        
        self._create_results_table("% Tiles Visited (Epsilon Decay Combinations)", row_labels_decay, col_labels_decay, percent_data_decay, "epsilon_decay_percent_visited.png")
        self._create_results_table("Avg Reward per Episode (Epsilon Decay Combinations)", row_labels_decay, col_labels_decay, reward_data_decay, "epsilon_decay_avg_reward.png")
        self._create_results_table("Avg Success Rate (Epsilon Decay Combinations)", row_labels_decay, col_labels_decay, success_data_decay, "epsilon_decay_success_rate.png")

        # for init_epsilon in init_epsilons:
        #     for end_epsilon in end_epsilons:
        #         for rate in decay_rates:
        #             percent_correct = results_decay[j][0]
        #             avg_reward = results_decay[j][1]
        #             avg_successful_episodes = results_decay[j][2]
        #             j += 1
        #             print(f"init_epsilon, end_epsilon, decay_rate: {init_epsilon}, {end_epsilon}, {rate}, Avg. Percent Correct: {percent_correct}, avg reward: {avg_reward}, avg successful episodes: {avg_successful_episodes}")

        # Test different gamma values
        # gammas = [0.5, 0.7, 0.9, 0.99, 1.0]
        # results_gamma = []
        # k = 0

        # for gamma in gammas:
        #     percent_correct, avg_reward, avg_successful_episodes = repeatExperiments(
        #         self, env, n_episodes=2000, step_size=0.1, epsilon=0.1, 
        #         end_epsilon=0.05, decay_rate=4000000, gamma=gamma
        #     )
        #     results_gamma.append((percent_correct, avg_reward, avg_successful_episodes))
        
        # # Create tables for gamma testing
        # percent_data = [[r[0] for r in results_gamma]]
        # reward_data = [[r[1] for r in results_gamma]]
        # success_data = [[r[2] for r in results_gamma]]
        
        # col_labels = [f"γ={g}" for g in gammas]
        # row_labels = ["Results"]
        
        # self._create_results_table("Percent Correct by Gamma", row_labels, col_labels, percent_data, "gamma_percent_correct.png")
        # self._create_results_table("Avg Reward by Gamma", row_labels, col_labels, reward_data, "gamma_avg_reward.png")
        # self._create_results_table("Avg Successful Episodes by Gamma", row_labels, col_labels, success_data, "gamma_avg_success.png")
        
        # # Also create a combined table with all 3 metrics
        # combined_data = [
        #     [r[0] for r in results_gamma],  # Percent correct
        #     [r[1] for r in results_gamma],  # Avg reward
        #     [r[2] for r in results_gamma]   # Avg successful episodes
        # ]
        # combined_row_labels = ["% Correct", "Avg Reward", "Avg Success"]
        
        # self._create_results_table("Gamma Testing Results (All Metrics)", combined_row_labels, col_labels, combined_data, "gamma_all_metrics.png")
        
        # for gamma in gammas:
        #     percent_correct = results_gamma[k][0]
        #     avg_reward = results_gamma[k][1]
        #     avg_successful_episodes = results_gamma[k][2]
        #     k += 1
        #     print(f"gamma: {gamma}, Avg. Percent Correct: {percent_correct}, avg reward: {avg_reward}, avg successful episodes: {avg_successful_episodes}")

    def deploy(self, render: bool = True, max_steps: int = 500):
        env = gym.make(self.env_id, level_str=self.level_str, render_mode="human" if render else None)

        #done training, want the results
        f = open(os.path.join(self.getPkFolderPath(self.algorithm_name), self.reference_name + '_solution.pk1'), "rb")
        q = pickle.load(f)
        f.close()

        #Reset env before each episode
        step_count = 0
        state = env.reset()[0]
        terminated = False #terminated just means found target

        while (not terminated and step_count < max_steps):
            step_count += 1

            # From the state value, get the available actions mask, which is an int where each bit represents an action
            available_actions_mask = env.unwrapped.get_available_actions_mask(state)

            available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

            # If there are no available actions, choose randomly from all actions
            if len(available_actions) == 0:
                available_actions = env.unwrapped.get_termination_actions(state)

            # Get the action with the highest Q-value that's also in available actions
            action = available_actions[np.argmax(q[state, available_actions])]

            # Perform the action
            new_state,reward,terminated,_,_ = env.step(action)
            
            # Update State
            state = new_state

        #for loop done
        env.close()
    

if __name__ == '__main__':
    agent: ThinIceQLearningAgent = ThinIceQLearningAgent('thin-ice-v1', 'level_6.txt')
    # For hyperparameter testing:
    agent.runQLearningTesting(n_runs=5)
    
    # For regular training:
    # agent.train(n_episodes=20000, step_size=0.1, gamma=1, epsilon=0.5)
    # agent.deploy(render=True)
    #agent.visualize_policy()
