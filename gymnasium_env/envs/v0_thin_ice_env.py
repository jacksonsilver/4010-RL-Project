'''
COMP4010 Project - Custom Gym Environment for Thin Ice 
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_thin_ice as ti
import numpy as np

register(
    id='thin-ice-v0', # unique id for the environment
    entry_point='gymnasium_env.envs:ThinIceEnv', # module_name:class_name
)

class ThinIceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, path_to_level='Textfiles\Level1.txt'):
        self.render_mode = render_mode

        # Initialize the level and player
        self.level = ti.Level(path_to_level)
        self.player = ti.ThinIcePlayer(self.level)

        # Define action and observation space
        self.action_space = spaces.Discrete(len(ti.PlayerActions))
        self.observation_space = spaces.Box(
            low=0, 
            high= np.array([self.level.get_num_rows()-1, self.level.get_num_cols()-1, self.level.get_num_rows()-1, self.level.get_num_cols()-1]),
            shape=(4,),
            dtype=np.int32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player.reset(seed)

        obs = np.concatenate((self.player.player_pos, self.player.target_pos))

        # Debug information
        info = {} 

        if self.render_mode == "human": 
            self.player.render()
        
        return obs, info

    def step(self, action):
        target_reached = self.player.perform_action(ti.PlayerActions(action))

        reward = 1 if target_reached else 0
        terminated = target_reached

        obs = np.concatenate((self.player.player_pos, self.player.target_pos))

        # Debug information
        info = {}

        if self.render_mode == "human":
            print(f"Action taken: {ti.PlayerActions(action)}")
            self.player.render()
        
        # Return observation, reward, done, truncated (not used), info
        return obs, reward, terminated, False, info
    
    def render(self):
        self.player.render()

# Run to test the environment
if __name__ == "__main__":
    env = gym.make('thin-ice-v0', render_mode='human', path_to_level='Textfiles\Level1.txt')

    print("Check environment begin")
    check_env(env, warn=True)
    print("Check environment end")

    # Reset the environment
    obs, info = env.reset()

    # Take a random action
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Reached the target!")
            break
        