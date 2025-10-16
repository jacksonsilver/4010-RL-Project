'''
COMP4010 Project - Custom Gym Environment for Thin Ice 
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_thin_ice as ti
import numpy as np
import pygame
import os

#Register -> to be able to use it as ID
register(
    id='thin-ice-v0', # unique id for the environment
    entry_point='gymnasium_env.envs:ThinIceEnv', # module_name:class_name
)

class ThinIceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, level_str='level_0.txt'):
        self.render_mode = render_mode

        # Initialize the level and player
        self.level = ti.Level(level_str)
        self.player = ti.ThinIcePlayer(self.level)

        # Define action and observation space
        self.action_space = spaces.Discrete(len(ti.PlayerActions))  #randomly select an action
        self.observation_space = spaces.Box(
            low=0, 
            high= np.array([self.level.get_num_cols()-1, self.level.get_num_rows()-1]),
            shape=(2,),
            dtype=np.int32
        )
        #displaying the grid in a window: human
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 32
            self.window_size = 600
            #till we need it!
            self.window = None
            self.clock = None

        #load tiles images
        self.tile_images = {}
        asset_path = "gymnasium_env/assets/"

        try:
            #map letters to images
            map_images = {
                'PF': 'floor_with_player.png',
                'PT': 'target_with_player.png',     
                'W': 'wall.webp',      
                'T': 'target.webp',        
                'F': 'floor.webp',  
                'B': 'blank.webp'
            }
                
            for tile, image_file in map_images.items():
                image_path = os.path.join(asset_path, image_file)
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    self.tile_images[tile] = pygame.transform.scale(image, (self.cell_size, self.cell_size))
                else:
                    print(f"Warning: Image file {image_path} not found.")
        except:
            self.tile_images = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player.reset(seed)

        obs = self.player.player_pos

        # Debug information
        info = {} 

        if self.render_mode == "human": 
            # self.player.render() #for terminal debugging
            self.render_pygame()
        return obs, info

    def step(self, action):
        target_reached = self.player.perform_action(ti.PlayerActions(action))

        reward = 1 if target_reached else 0
        terminated = target_reached

        obs = self.player.player_pos

        # Debug information
        info = {}

        if self.render_mode == "human":
            print(f"Action taken: {ti.PlayerActions(action)}")
            self.render_pygame() 
        
        # Return observation, reward, done, truncated (not used [eg: after 200 steps stop]), info
        return obs, reward, terminated, False, info
    
    def render(self):
        # self.player.render()
        if self.render_mode == "human":
            self.render_pygame()


    def render_pygame(self):
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()
        #surface to draw the tiles on
        surface = pygame.Surface((self.window_size, self.window_size))
        
        #draw the tiles for the leavez
        for i in range(self.level.get_num_rows()):
            for j in range(self.level.get_num_cols()):
                tile = self.level.get_tile((j, i))
                if tile is not None:
                    tile_char = str(tile)
                    x = j * self.cell_size
                    y = i * self.cell_size
                    
                    # Draw tile image 
                    if tile_char in self.tile_images:
                        surface.blit(self.tile_images[tile_char], (x, y))
       
        # Draw penguin to move
        player_x, player_y = self.player.player_pos
        px = player_x * self.cell_size
        py = player_y * self.cell_size
        
        if self.level.get_tile(self.player.player_pos).tile_type == ti.LevelTileType.TARGET:
            surface.blit(self.tile_images['PT'], (px, py))
        else:  
            surface.blit(self.tile_images['PF'], (px, py))
       
        # Update display(copy surface to window)
        self.window.blit(surface, (0, 0))
        #make sure the display is visible
        pygame.display.flip()
        self.clock.tick(4) #FPS using 4 instead of 1 to make it faster


# Run to test the environment
if __name__ == "__main__":
    env = gym.make('thin-ice-v0', render_mode='human', level_str='level_3.txt')

    print("======================================== Check environment begin ========================================")
    check_env(env, warn=True)
    print("========================================  Check environment end  ========================================")

    # Reset the environment
    obs, info = env.reset()

    print("======================================== START ROUND ========================================")
    # Take a random action
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Reached the target!")
            break
        