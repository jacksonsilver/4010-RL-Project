import os
from v0_qlearning import ThinIceQLearningAgent
from v1_thin_ice_PPO import ThinIcePPOAgent

import contextlib

def getConsolePath(level_number):
    return os.path.join(os.path.dirname(__file__),"console_outputs","PPO",f"ppo_training_level_{level_number}.txt")

def QLearning(n_episodes,step_size,gamma,epsilon,level_number,train):
    agent = ThinIceQLearningAgent('thin-ice-v0', f'level_{level_number}.txt')

    if train:
        agent.train(gamma,step_size,epsilon,n_episodes)
    else:
        agent.deploy(render=True)
        agent.visualize_policy()

def all_levels(gamma,learning_rate):
    
    #1,2,3
    '''
    worked with reward +1 and FINLA REWARD OF -1 
    entrop = 0.1
    '''
    # for i in range(1,4):
    #     print(f'cURRENTLY LEVEL {i}')
    #     PPO(gamma,learning_rate,100,i,train = True,saveToLog=True)  #train
    #     PPO(gamma,learning_rate,100,i,train = False)  #deploy
    

    # #4,5
    # for i in range(4,7):
    #     print(f'cURRENTLY LEVEL {i}')
    #     PPO(gamma,learning_rate,800,i,train = True,saveToLog=True)  #train
    #     PPO(gamma,learning_rate,800,i,train = False)  #deploy

    # 6,7,8,9
    # for i in range(6,10):
    #     print(f'cURRENTLY LEVEL {i}')
    #     PPO(gamma,learning_rate,2000,i,train = True,saveToLog=True)  #train
    #     PPO(gamma,learning_rate,2000,i,train = False)  #deploy

    # #10,11
    # for i in range(10,12):
    #     print(f'cURRENTLY LEVEL {i}')
    #     PPO(gamma,learning_rate,1500,i,train = True,saveToLog=True)  #train
    #     PPO(gamma,learning_rate,1500,i,train = False)  #deploy


def PPO(gamma,lr,n_episodes,level_number,train,saveToLog  = False):
    agent = ThinIcePPOAgent('thin-ice-v0', f'level_{level_number}.txt')

    if train:
        if saveToLog:
            print("Training.....")
            log_path = getConsolePath(level_number)

            with open(log_path, "w") as log_file:
                with contextlib.redirect_stdout(log_file):
                    agent.train(gamma, lr, n_episodes)
        else:
            agent.train(gamma, lr, n_episodes)

    else:
        print("Deploying.....")
        agent.deploy(render=True)

if __name__ == '__main__':

    #Parameters being used
    n_episodes = 500 #25000 for qlearning, 2000 for PPO
    #3000  for level 4
    #3500  for level 5
    #

    step_size = 0.1
    gamma = 1
    epsilon = 0.1
    learning_rate = 0.0003

    #Level we are working with
    level_number = 6

    PPO(gamma,learning_rate,n_episodes,level_number,train = True,saveToLog=True)  #train
    PPO(gamma,learning_rate,5,level_number,train = False)  #deploy
    
    '''Algorithms currently implemented'''

    #QLearning(n_episodes,step_size,gamma,epsilon,level_number,train = True)
    #QLearning(n_episodes,step_size,gamma,epsilon,level_number,train = False)

