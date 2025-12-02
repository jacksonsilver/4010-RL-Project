import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import random

from thin_ice_training_agent import ThinIceTrainingAgent
import v1_thin_ice_env as ti
from components.decaying_epsilon import DecayingEpsilon

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.embed = nn.Embedding(in_states, h1_nodes)
        self.fc1 = nn.Linear(h1_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, state_ids):
        x = self.embed(state_ids)
        x = F.relu(self.fc1(x))
        return self.out(x)

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen, state_shape, action_shape=(), dtype=torch.int32):
        self.maxlen = maxlen
        self.ptr = 0          # current insert position
        self.size = 0         # number of stored transitions
        
        # Preallocate arrays
        self.states = torch.zeros((maxlen, *state_shape), dtype=dtype)
        self.actions = torch.zeros((maxlen, *action_shape), dtype=dtype)
        self.rewards = torch.zeros((maxlen,), dtype=torch.int32)
        self.next_states = torch.zeros((maxlen, *state_shape), dtype=dtype)
        self.dones = torch.zeros((maxlen,), dtype=torch.int32)

    def append(self, state, action, reward, next_state, done):
        i = self.ptr

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done

        self.ptr = (self.ptr + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))

        return ( self.states[idx],
                 self.actions[idx],
                 self.rewards[idx],
                 self.next_states[idx],
                 self.dones[idx] )

    def __len__(self):
        return self.size


class ThinIceDQLAgent(ThinIceTrainingAgent):
    # Hyperparameters (adjustable)
    discount_factor_g = 1         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    def __init__(self, env_id: str ='thin-ice-v1', level_str: str ='level_1.txt'):
        super().__init__("DQL", env_id, level_str)
    
    # Train the Thin Ice environment
    def train(self, epsilon=0.1, step_size = 0.001, n_episodes=1000):
        # Create Thin Ice instance
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        memory = ReplayMemory(
            maxlen=self.replay_memory_size,
            state_shape=(),
            action_shape=()
        )

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=step_size)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(n_episodes)
        epsilon_history = np.zeros(n_episodes)

        decaying_epsilon = DecayingEpsilon(start_epsilon=epsilon, end_epsilon=0.1, decay_rate=50000)

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in range(n_episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            print(f"Episode {i+1}/{n_episodes}")

            while(not terminated and not truncated):
                # From the state value, get the available actions mask, which is an int where each bit represents an action
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                # If there are no available actions, choose randomly from an action that will terminate the episde
                if len(available_actions) == 0:
                    available_actions = env.unwrapped.get_termination_actions(state)

                # Select action based on epsilon-greedy
                if np.random.rand() < decaying_epsilon.get_epsilon():
                    # select random action
                     action = np.random.choice(available_actions)
                else:
                    # select best action            
                    with torch.no_grad():
                        q_values = policy_dqn(self.state_to_dqn_input(state, num_states))

                    q_values = q_values[0]  # flatten to shape [num_actions]

                    # Create 1D mask of same length
                    available_mask = torch.zeros(q_values.shape[0], dtype=torch.bool)
                    available_mask[available_actions] = True

                    # Mask out unavailable actions
                    q_values[~available_mask] = -1e15
                    action = torch.argmax(q_values).item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append(state, action, reward, new_state, terminated) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            rewards_per_episode[i] = reward

            # Track epsilon
            epsilon_history[i] = decaying_epsilon.get_epsilon()

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                decaying_epsilon.update(episode=i)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "thin_ice_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(n_episodes)
        for x in range(n_episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel("Episode Number")
        plt.ylabel("Reward per Episode")
        plt.title("Rewards Over Episodes")
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        plt.xlabel("Episode Number")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Episodes")
        
        # Save plots
        plt.tight_layout()
        plt.savefig('thin_ice_dql.png')

        return policy_dqn.state_dict(), rewards_per_episode
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        states, actions, rewards, next_states, dones = mini_batch

        q_current = policy_dqn(self.state_batch_to_dqn_input(states, num_states))  

        with torch.no_grad():
            q_next = target_dqn(self.state_batch_to_dqn_input(next_states, num_states))  
            max_q_next = q_next.max(dim=1).values


        targets = rewards + self.discount_factor_g * max_q_next * (1 - dones)

        q_target_full = q_current.clone()
        q_target_full[torch.arange(len(actions)), actions] = targets
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(q_current, q_target_full)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state, num_states)->torch.Tensor:
        '''
        Converts an state (int) to a tensor representation.
        '''
        return torch.tensor([state], dtype=torch.long)

    
    def state_batch_to_dqn_input(self, state_batch, num_states):
        return state_batch.detach().clone().long()

    # Run the FrozeLake environment with the learned policy
    def deploy(self, episodes, max_steps=100):
        # Create FrozenLake instance
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("thin_ice_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.run_policy(policy_dqn=policy_dqn, env=env, episodes=episodes, max_steps=max_steps)
    
    def run_policy(self, policy_dqn, env, episodes=1, max_steps=100):
        num_states = env.observation_space.n

        visited_tile_percents = []

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False
            truncated = False

            total_visitable_tiles = env.unwrapped.level.get_visitable_tile_count()

            steps = 0
            while(not terminated and not truncated):  
                # Select best action   
                # From the state value, get the available actions mask, which is an int where each bit represents an action
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                # If there are no available actions, choose randomly from all actions
                if len(available_actions) == 0:
                    available_actions = env.unwrapped.get_termination_actions(state)

                # select best action            
                with torch.no_grad():
                    q_values = policy_dqn(self.state_to_dqn_input(state, num_states))

                q_values = q_values[0]  # flatten to shape [num_actions]

                # Create 1D mask of same length
                available_mask = torch.zeros(q_values.shape[0], dtype=torch.bool)
                available_mask[available_actions] = True

                # Mask out unavailable actions
                q_values[~available_mask] = -1e15

                action = torch.argmax(q_values).item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

                steps += 1
                # End the episode if it's taking too long (likely didn't train properly)
                if steps >= max_steps:
                    truncated = True
            
            visited_tiles = env.unwrapped.level.get_visited_tile_count()
            visited_tile_percents.append((visited_tiles / total_visitable_tiles) * 100)

        env.close()
        return visited_tile_percents

    def runDQLtesting(self):
        def repeatExperiments(self, env, episode, step_size, epsilon):
            num_states = env.observation_space.n
            num_actions = env.action_space.n
            total_visit_percents = []
            for i in range(n_runs):
                print("TRAINING")
                policy_dict, rewards_per_episode = self.train(n_episodes=episode, epsilon=epsilon, step_size=step_size)

                # Load learned policy
                policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
                policy_dqn.load_state_dict(policy_dict)
                policy_dqn.eval()    # switch model to evaluation mode

                print("EVALUATE")
                visited_tile_percents = self.run_policy(policy_dqn=policy_dqn, env=env)
                total_visit_percents.extend(visited_tile_percents)
            
            return sum(total_visit_percents) / len(total_visit_percents)


        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str, render_mode=None)

        # Go through different episode counts and step sizes to see performance
        step_sizes = [0.001, 0.0001]
        episode_counts = [500, 1000, 2000]
        init_epsilons = [0.1, 0.3, 0.5]
        n_runs = 1

        results = []
        i = 0

        for episode in episode_counts:
            for step_size in step_sizes:
                for epsilon in init_epsilons:
                    percent_correct = repeatExperiments(self, env, episode, step_size, epsilon)
                    results.append(percent_correct)

        for episode in episode_counts:    
            for step_size in step_sizes:
                for epsilon in init_epsilons:
                    percent_correct = results[i]
                    i += 1
                    print(f"alpha, episode, epsilon: {step_size}, {episode}, {epsilon}, Avg. Percent Correct: {percent_correct}")
        

if __name__ == '__main__':
    thin_ice_agent = ThinIceDQLAgent(env_id="thin-ice-v1", level_str='level_6.txt')
    thin_ice_agent.train(epsilon=0.5, step_size = 0.001, n_episodes=1000)
    thin_ice_agent.deploy(episodes=3)

    # LARGE TESTING
    # thin_ice_agent.runDQLtesting()