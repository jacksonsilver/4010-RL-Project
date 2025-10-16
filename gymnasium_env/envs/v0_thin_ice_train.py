import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
#import stable.baseliens3 import A2C
import v0_thin_ice_env #including it so it registers 
import os

from thin_ice_training_agent import ThinIceTrainingAgent

class ThinIceQLearningAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='Level0.txt'):
        super().__init__(env_id, level_str)

    def train(self, n_episodes: int = 100):
        env = gym.make(self.env_id, level_str=self.level_str)

        # Table Shape is Number of Cols (X) * Num of Rows (Y) * Number of Actions 
        q = np.zeros((env.unwrapped.level.get_num_cols(),
            env.unwrapped.level.get_num_rows(),
            env.action_space.n))

        #Hyperparameters
        alpha = 0.9 
        discount_factor = 0.9 #can also be called gamma
        epsilon = 1  #100% random actions

        #keeping count of number of stesp per episode (is it becoming more efficient or not)
        number_of_steps = np.zeros(n_episodes)

        step_count = 0
        for i in range(n_episodes):
            print(f'Episode: {i}')

            #Reset env before each episode
            state = env.reset()[0]
            terminated = False #terminated just means found target

            while (not terminated):
                
                #picking an action based on episoln greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    #thing we learned in class
                    q_state_index = tuple(state)
                    action = np.argmax(q[q_state_index])

                new_state,reward,terminated,_,_ = env.step(action)

                q_state_action_index = tuple(state) + (action,) # Creates index of X,Y,Action
                q_new_state_index = tuple(new_state) # Generic index of X',Y', where X',Y' is the new position after Action

                #Update q table with formula from class
                q[q_state_action_index] = q[q_state_action_index] + alpha * (
                        reward + discount_factor * np.max(q[q_new_state_index]) - q[q_state_action_index]
                )
                
                #Update State
                state = new_state

                step_count += 1
                if terminated:
                    #which means the goal was reached, keep note of step #
                    number_of_steps[i] = step_count
                    step_count = 0
            
            #decrease epsilon but why?
            epsilon = max(epsilon - 1/n_episodes,0)

        #for loop done
        env.close()

        self.generate_graph(number_of_steps)

        f  = open(self.reference_name + '_solution.pk1',"wb")
        pickle.dump(q,f)
        f.close()

    def deploy(self, render: bool = True):
        env = gym.make(self.env_id, level_str=self.level_str, render_mode="human" if render else None)

        #done training, want the results
        f = open(self.reference_name + '_solution.pk1','rb')
        q = pickle.load(f)
        f.close()

        #Reset env before each episode
        state = env.reset()[0]
        terminated = False #terminated just means found target

        while (not terminated):
            # Get index based on state
            q_state_index = tuple(state)

            # Get action from Q table based on state
            action = np.argmax(q[q_state_index])

            # Perform the action
            new_state,reward,terminated,_,_ = env.step(action)
            
            # Update State
            state = new_state

        #for loop done
        env.close()
    

if __name__ == '__main__':
    agent: ThinIceQLearningAgent = ThinIceQLearningAgent('thin-ice-v0', 'level_6.txt')
    agent.train(500)
    agent.deploy(render=True)






                      


