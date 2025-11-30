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

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


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
    
    # Train the FrozeLake environment
    def train(self, epsilon=0.1, step_size = 0.001, n_episodes=1000):
        # Create FrozenLake instance
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=step_size)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(n_episodes)

        decaying_epsilon = DecayingEpsilon(start_epsilon=epsilon, end_epsilon=0.1, decay_rate=500000)

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
                    # with torch.no_grad():
                    #     action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                    with torch.no_grad():
                        q_values = policy_dqn(self.state_to_dqn_input(state, num_states))

                    # Mask out unavailable actions by setting them to -inf
                    masked_q = q_values.clone()
                    for a in range(len(q_values)):
                        if a not in available_actions:
                            masked_q[a] = -1e15  # effectively removes action

                    action = torch.argmax(masked_q).item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            rewards_per_episode[i] = reward

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

        return policy_dqn.state_dict(), rewards_per_episode
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        '''
        Converts an state (int) to a tensor representation.
        '''
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

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
                # with torch.no_grad():
                #     action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                with torch.no_grad():
                    q_values = policy_dqn(self.state_to_dqn_input(state, num_states))

                # Mask out unavailable actions by setting them to -inf
                masked_q = q_values.clone()
                for a in range(len(q_values)):
                    if a not in available_actions:
                        masked_q[a] = -1e15  # effectively removes action

                action = torch.argmax(masked_q).item()

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
        n_runs = 3

        results = []
        i = 0

        for step_size in step_sizes:
            for episode in episode_counts:
                for epsilon in init_epsilons:
                    percent_correct = repeatExperiments(self, env, episode, step_size, epsilon)
                    results.append(percent_correct)
        
        for step_size in step_sizes:
            for episode in episode_counts:
                for epsilon in init_epsilons:
                    percent_correct = results[i]
                    i += 1
                    print(f"alpha, episode, epsilon: {step_size}, {episode}, {epsilon}, Avg. Percent Correct: {percent_correct}")
        

if __name__ == '__main__':
    thin_ice_agent = ThinIceDQLAgent(env_id="thin-ice-v1", level_str='level_6.txt')
    thin_ice_agent.train(epsilon=0.1, step_size = 0.001, n_episodes=1000)
    thin_ice_agent.deploy(episodes=3)

    # LARGE TESTING
    #thin_ice_agent.runDQLtesting()