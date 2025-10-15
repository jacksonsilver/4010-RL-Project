import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
#import stable.baseliens3 import A2C
import v0_thin_ice_env #including it so it registers 
import os


### Using Q-Learning

def train_q_learning(episodes,training = True,render = False): #episodes just means how many times

    env = gym.make('thin-ice-v0',render_mode ='human' if render else None)


    if(training):
        q = np.zeros((env.unwrapped.level.get_num_rows(),
              env.unwrapped.level.get_num_cols(),
              env.unwrapped.level.get_num_rows(),
              env.unwrapped.level.get_num_cols(),
              env.action_space.n))
    else:
        #done training, want the results
        f = open('v0_thin_ice_solution.pk1','rb')
        q = pickle.load(f)
        f.close()

    #Hyperparameters
    alpha = 0.9 
    discount_factor = 0.9 #can also be called gamma
    epsilon = 1  #100% random actions

    #keeping count of number of stesp per episode (is it becoming more efficient or not)
    number_of_steps = np.zeros(episodes)

    step_count = 0
    for i in range(episodes):
        if(render):
            print(f'Epsiode: {i}')

        #Reset env before each episode
        state = env.reset()[0]
        terminated = False #terminated just means found target

        while (not terminated):
            
            #picking an action based on episoln greedy
            if training and random.random() < epsilon:
                action = env.action_space.sample()
            else:
                #thing we learned in class
                q_state_index = tuple(state)
                action = np.argmax(q[q_state_index])

            new_state,reward,terminated,_,_ = env.step(action)

            #not sure whats happening here tbh
            q_state_action_index = tuple(state) + (action,)
            q_new_state_index = tuple(new_state)

            if training:
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
        epsilon = max(epsilon - 1/episodes,0)

    #for loop done
    env.close()

    #graph everything where x = epsiodes and y = number of steps 
    ''' notice in graph that first epsiodes have greater # of steps, still in training'''
    sum_steps  = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(number_of_steps[max(0,t-100):(t+1)]) #avg step
    plt.plot(sum_steps)
    path_for_graph = os.path.join(os.path.dirname(__file__), 'graphs_generated', 'v0_thin_ice_solution.png')
    plt.savefig(path_for_graph)

    if training:
        f  = open('v0_thin_ice_solution.pk1',"wb")
        pickle.dump(q,f)
        f.close()



if __name__ == '__main__':
    train_q_learning(500,training=True,render=False) #for testing, keeping 2 as small number
    train_q_learning(1,training=False,render=True)  






                      


