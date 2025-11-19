import gymnasium as gym
import numpy as np
import pickle
import v0_thin_ice_env as ti #including it so it registers 
import os

from thin_ice_training_agent import ThinIceTrainingAgent
from components.replay_buffer import ReplayBuffer

class ThincIceDynaQAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='Level0.txt'):
        super().__init__("DynaQ", env_id, level_str)
    
    def train(self, gamma=0.99, step_size=0.1, epsilon=0.1, n_episodes=1000, max_model_step=10):
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)

        # Initialize Q table with random values for n_states x n_actions
        q = np.random.rand(env.unwrapped.n_states,
            env.unwrapped.n_actions)
        # Ensure that goal (terminal) state(s) is initialized to 0
        targets = env.unwrapped.target
        q[targets, :] = 0

        model = ReplayBuffer(step_size, gamma, max_step=max_model_step)

        number_of_steps = np.zeros(n_episodes)

        for i in range(n_episodes):
            print(f'Episode: {i}')

            state = env.reset()[0]
            terminated = False
            step_count = 0

            while not terminated:
                step_count += 1

                # Choose action from state based on epsilon-greedy policy
                if np.random.rand() < epsilon:
                    # From the state value, get the available actions mask, which is an int where each bit represents an action
                    available_actions_mask = env.unwrapped.get_available_actions_mask(state)

                    available_actions = env.unwrapped.action_mask_to_actions(available_actions_mask)

                    # If there are no available actions, choose randomly from all actions
                    if len(available_actions) == 0:
                        available_actions = list(range(env.unwrapped.n_actions))
                    
                    action = np.random.choice(available_actions)
                else:
                    action = np.argmax(q[state])
                
                next_state, reward, terminated, truncated, _ = env.step(action)

                if truncated:
                    terminated = True

                if terminated:
                    q_target = reward
                else:
                    next_action = np.argmax(q[next_state])
                    q_target = reward + gamma * q[next_state, next_action]
                
                q[state, action] = q[state, action] + step_size * (q_target - q[state, action])

                model.UpdateExperiences(state, action, reward, next_state)

                model.UpdateQ(q)

                state = next_state
            
            number_of_steps[i] = step_count

        policy = np.zeros((env.n_states, env.n_actions))
        for i in range(env.n_states):
            best_action = np.argmax(q[i])
            policy[i, best_action] = 1.0
        
        #for loop done
        env.close()

        self.generate_graph(number_of_steps)

        f = open(os.path.join(self.getPkFolderPath(self.algorithm_name), self.reference_name + '_solution.pk1'), "wb")
        pickle.dump(q,f)
        f.close()

        return policy, q.reshape(-1, 1)

    def deploy(self, render: bool = False, max_steps: int = 500):
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

            # Get action from Q table based on state
            action = np.argmax(q[state])

            # Perform the action
            new_state,reward,terminated,_,_ = env.step(action)
            
            # Update State
            state = new_state

        #for loop done
        env.close()

if __name__ == '__main__':
    agent = ThincIceDynaQAgent('thin-ice-v0', 'level_6.txt')
    agent.train(n_episodes=1000, step_size=0.1, gamma=1, epsilon=0.1)
    agent.deploy(render=True)