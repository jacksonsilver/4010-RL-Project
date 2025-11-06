import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
#import stable.baseliens3 import A2C
import v0_thin_ice_env as ti #including it so it registers 
import os

from thin_ice_training_agent import ThinIceTrainingAgent

class ThinIceQLearningAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='Level0.txt'):
        super().__init__(env_id, level_str)

    def train(self, gamma: float = 0.9, step_size: float = 0.1, epsilon: float = 0.1, n_episodes: int = 1000):
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)

        # Initialize Q table with random values for n_states x n_actions
        q = np.random.rand(env.unwrapped.n_states,
            env.unwrapped.n_actions)
        # Ensure that goal (terminal) state(s) is initialized to 0
        targets = env.unwrapped.target
        q[targets, :] = 0

        #keeping count of number of stesp per episode (is it becoming more efficient or not)
        number_of_steps = np.zeros(n_episodes)

        for i in range(n_episodes):
            print(f'Episode: {i}')

            #Reset env before each episode
            state = env.reset()[0]
            step_count = 0
            terminated = False #terminated just means found target

            while (not terminated):
                step_count += 1

                # Choose action from state based on epsilon-greedy policy
                if np.random.rand() < epsilon:
                    # From the state value, get the available actions mask, which is an int where each bit represents an action
                    available_actions_mask = env.unwrapped._to_cell[state][3]

                    available_actions = []
                    # Iterate over action bits (bit i corresponds to action i)
                    for i_action in range(env.unwrapped.n_actions):
                        if (available_actions_mask >> i_action) & 1: # Means that this action is == 1 and is available
                            available_actions.append(i_action)

                    # If there are no available actions, choose randomly from all actions
                    if len(available_actions) == 0:
                        available_actions = list(range(env.unwrapped.n_actions))
                    
                    action = np.random.choice(available_actions)
                else:
                    action = np.argmax(q[state])

                next_state,reward,terminated,_,_ = env.step(action)


                # Update q table with formula from class
                next_action = np.argmax(q[next_state])
                q[state, action] = q[state, action] + step_size * (
                        reward + gamma * np.max(q[next_state, next_action]) - q[state, action]
                )
                
                # Update State
                state = next_state
            
            number_of_steps[i] = step_count

        #for loop done
        env.close()

        self.generate_graph(number_of_steps)

        f = open(os.path.join(self.getPkFolderPath(), self.reference_name + '_solution.pk1'), "wb")
        pickle.dump(q,f)
        f.close()

    def deploy(self, render: bool = True, max_steps: int = 500):
        env = gym.make(self.env_id, level_str=self.level_str, render_mode="human" if render else None)

        #done training, want the results
        f = open(os.path.join(self.getPkFolderPath(), self.reference_name + '_solution.pk1'), "rb")
        q = pickle.load(f)
        f.close()

        #Reset env before each episode
        step_count = 0
        state = env.reset()[0]
        terminated = False #terminated just means found target

        while (not terminated and step_count < max_steps):
            step_count += 1

            # Get action from Q table based on state
            action = np.argmax(q[state])

            # Perform the action
            new_state,reward,terminated,_,_ = env.step(action)
            
            # Update State
            state = new_state

        #for loop done
        env.close()


