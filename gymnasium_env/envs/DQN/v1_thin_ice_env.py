'''
COMP4010 Project - Custom Gym Environment for Thin Ice 
'''
import random
from typing import Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from . import v1_thin_ice as ti
import numpy as np
import pygame
import os

#Register -> to be able to use it as ID
register(
    id='thin-ice-v1', # unique id for the environment
    entry_point='gymnasium_env.envs:ThinIceEnv', # module_name:class_name
)


class ThinIceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, level_str: str = None, level_list: list[str] = None, max_rows: int = 15, max_cols: int = 19):
        super().__init__()
        self.render_mode = render_mode

        self.level_list = level_list
        self.single_level_str = level_str

        self.max_rows = max_rows
        self.max_cols = max_cols

        self._level = None
        # Define number of actions and states
        self._n_actions = len(ti.PlayerActions)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)  #randomly select an action
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(low=0.0, high=1.0, shape=(5, self.max_rows, self.max_cols), dtype=np.float32),
            'action_mask': spaces.MultiBinary(self._n_actions)
        })

        # Set visited tiles parameter
        self.visited_tiles = set()
        self.cell_size = 32

        # Load tiles images
        self.tile_images = {}
        asset_path = "gymnasium_env/assets/"

        # Displaying the grid in a window: human
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 32
            self.window_size = 600
            #till we need it!
            self.window = None
            self.clock = None

        try:
            #map letters to images
            map_images = {
                'PF': 'floor_with_player.png',
                'PT': 'target_with_player.png',     
                'PW': 'water_with_player.png',
                'W': 'wall.webp',      
                'T': 'target.webp',        
                'F': 'floor.webp',  
                'B': 'blank.webp',
                '~': 'water.png',
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


    def _get_obs_dict(self):
        # Initialize observation and action grids
        grid_obs = np.zeros((5, self.max_rows, self.max_cols), dtype=np.float32)
        action_mask_arr = np.zeros(self._n_actions, dtype=np.int8)

        # read level into convolutional layers so that agent can see env
        if self._level is not None:
            rows, cols = self._level.n_rows, self._level.n_cols
            px, py = self._level.player_position

            if 0 <= py < self.max_rows and 0 <= px < self.max_cols: 
                grid_obs[0, py, px] = 1.0

            for r in range(min(rows, self.max_rows)):
                for c in range(min(cols, self.max_cols)):
                    tile = self._level.get_tile((c, r));
                    if tile is None: 
                        continue
                    tt = tile.tile_type
                    if tt == ti.LevelTileType.WALL: 
                        grid_obs[1, r, c] = 1.0
                    elif tt == ti.LevelTileType.WATER: 
                        grid_obs[2, r, c] = 1.0
                    elif tt == ti.LevelTileType.TARGET: 
                        grid_obs[3, r, c] = 1.0
                    elif tt == ti.LevelTileType.FLOOR: 
                        grid_obs[4, r, c] = 1.0

            current_tile = self._level.player_position_tile
            if current_tile is not None:
                mask= self._level.get_available_actions(current_tile)
                for i in range(self._n_actions):
                    if (mask >> i) & 1: action_mask_arr[i] = 1

        return {'obs': grid_obs, 'action_mask': action_mask_arr}

    def reset(self, seed=None, options=None):
        # Reset environment, level, and visited tiles
        super().reset(seed=seed)

        # Agent restarts learning from random list
        level_to_load = random.choice(self.level_list) if self.level_list else self.single_level_str

        self._level = ti.Level(level_to_load)

        self.visited_tiles = {self._level.player_position}

        obs_dict = self._get_obs_dict()

        info = {}
        if self.render_mode == "human": 
            self.render()
        return obs_dict, info

    def step(self, action):
        
        # Perform action
        target_reached, pos_changed = self.level.perform_action(ti.PlayerActions(action))
        new_pos = self._level.player_position

        player_tile = self._level.get_tile(new_pos)
        terminated, reward, avail_mask = False, 0, 0
        info = {}

        # Make sure curr position is valid
        if player_tile is None: 
            reward = -1
            terminated = True

        # Going into water results in death
        elif player_tile.tile_type == ti.LevelTileType.WATER: 
            reward = -1
            terminated = True 
        
        # Check that player finished game correctly (visited all tiles)
        elif target_reached:
            self.visited_tiles.add(new_pos)
            reward = 1 if len(self.visited_tiles) >= self._level.n_visitable_tiles else -1
            terminated = True

        if not terminated:
            # Punish no exploration
            if not pos_changed: 
                reward = -1

            else:
                # Stepped on a new tile -> More discovery -> Small Reward
                if new_pos not in self.visited_tiles: 
                    self.visited_tiles.add(new_pos)
                    reward = 1 
                # Should never be reached, but just in case of possible exploit
                else: 
                    reward = -1
                    terminated = True 

        # Check if agent trapped itself on a lone tile (no valid moves) and immediately terminate 
        if player_tile is not None and not terminated:
            avail_mask = self._level.get_available_actions(player_tile)
            if avail_mask == 0 and not target_reached: 
                reward = -1
                terminated = True

        obs_dict = self._get_obs_dict()
        info = {}

        truncated = False
        return obs_dict, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.render_pygame()

    def render_pygame(self):
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Pump events so the window remains responsive
        pygame.event.pump()

        # surface to draw the tiles on (reuse if desired)
        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill((255, 255, 255))

        # draw the tiles for the level
        for i in range(self.level.n_rows):
            for j in range(self.level.n_cols):
                tile = self.level.get_tile((j, i))
                if tile is not None:
                    tile_char = str(tile)
                    x = int(j * self.cell_size)
                    y = int(i * self.cell_size)

                    # Draw tile image if available
                    img = self.tile_images.get(tile_char)
                    if img is not None:
                        surface.blit(img, (x, y))

        # Draw penguin to move (use integer positions)
        ppos = self.level.player_position
        player_x, player_y = int(ppos[0]), int(ppos[1])
        px = int(player_x * self.cell_size)
        py = int(player_y * self.cell_size)

        # choose penguin image (target vs floor)
        current_tile = self.level.get_tile((player_x, player_y))
        if current_tile is not None:
            if current_tile.tile_type == ti.LevelTileType.TARGET:
                peng_img = self.tile_images.get('PT')
            elif current_tile.tile_type == ti.LevelTileType.WATER:
                peng_img = self.tile_images.get('PW')
            else:
                peng_img = self.tile_images.get('PF')

        if peng_img is not None:
            surface.blit(peng_img, (px, py))

        # Update display (copy surface to window)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # cap framerate using metadata
        fps = self.metadata.get('render_fps', 4)
        self.clock.tick(fps)

    def close(self):
        if self.window is not None:
            print("Closing Pygame...");
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            print("Pygame closed.")

    @property
    def level(self) -> ti.Level:
        return self._level
    
    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_states(self):
        return self._n_states
    
    @property
    def target(self):
        return self._target

    @property
    def to_cell(self):
        return self._to_cell
