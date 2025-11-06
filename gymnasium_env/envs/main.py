from v0_thin_ice_QLearning import ThinIceQLearningAgent
#from v0_thin_ice_PPO import ThinIcePPOAgent

def QLearning(n_episodes,step_size,gamma,epsilon,level_number,train):
    agent = ThinIceQLearningAgent('thin-ice-v0', f'level_{level_number}.txt')

    if train:
        agent.train(gamma,step_size,epsilon,n_episodes)
    else:
        agent.deploy(render=True)
        agent.visualize_policy()

# def PPO(gamma,lr,n_episodes,level_number,train):
#     agent = ThinIcePPOAgent('thin-ice-v0', f'level_{level_number}.txt')

#     if train:
#         agent.train(gamma,lr, n_episodes)
#     else:
#         agent.deploy(render=True)

if __name__ == '__main__':

    #Parameters being used
    n_episodes = 500  #25000 for qlearning, 2000 for PPO
    step_size = 0.1
    gamma = 1
    epsilon = 0.1
    learning_rate = 0.0001

    #Level we are working with
    level_number = 1


    '''Algorithms currently implemented'''

    #QLearning(n_episodes,step_size,gamma,epsilon,level_number,True)
    #QLearning(n_episodes,step_size,gamma,epsilon,level_number,False)


    #PPO(gamma,learning_rate,n_episodes,level_number,True)  #train
    #PPO(gamma,learning_rate,n_episodes,level_number,False)  #deploy

