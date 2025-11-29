import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import random

from thin_ice_training_agent import ThinIceTrainingAgent
import v0_thin_ice_env as ti

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

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
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 1         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='level_1.txt'):
        super().__init__("DQL", env_id, level_str)
    
    # Train the FrozeLake environment
    def train(self, epsilon=0.1, n_episodes=1000):
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
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(n_episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in range(n_episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            print(f"Episode {i+1}/{n_episodes}", end='\r')

            while(not terminated and not truncated):
                # From the state value, get the available actions mask, which is an int where each bit represents an action
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                # If there are no available actions, choose randomly from all actions
                if len(available_actions) == 0:
                    available_actions = list(range(env.unwrapped.n_actions))

                # Select action based on epsilon-greedy
                if np.random.rand() < epsilon:
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
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                # epsilon = max(epsilon - 1/n_episodes, 0)
                # epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "thin_ice_dql.pt")
    
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
    def deploy(self, episodes, max_steps=50):
        # Create FrozenLake instance
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("thin_ice_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False
            truncated = False

            steps = 0
            while(not terminated and not truncated):  
                # Select best action   
                # From the state value, get the available actions mask, which is an int where each bit represents an action
                available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                # If there are no available actions, choose randomly from all actions
                if len(available_actions) == 0:
                    available_actions = list(range(env.unwrapped.n_actions))

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

        env.close()

if __name__ == '__main__':
    thin_ice_agent = ThinIceDQLAgent(level_str='level_6.txt')
    thin_ice_agent.train(n_episodes=1000, epsilon=0.1)
    thin_ice_agent.deploy(episodes=3)