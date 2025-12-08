'''
COMP4010 Project - Custom Gym Environment for Thin Ice 
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.envs.registration import registry
from gymnasium.utils.env_checker import check_env

import gymnasium_env.envs.v0_thin_ice as ti
import numpy as np
import pygame
import os

#Register -> to be able to use it as ID
if 'thin-ice-v1' not in registry:
    register(
        id='thin-ice-v1', # unique id for the environment
        entry_point='gymnasium_env.envs:ThinIceEnv', # module_name:class_name
    )

class ThinIceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None, level_str='level_0.txt'):
        self.render_mode = render_mode

        # Initialize the level
        self._level = ti.Level(level_str)
        self._target = []

        # Set visited tiles parameter
        self.visited_tiles = set()

        state_num = 0
        self._to_state = {}

        # Build mapping from (x,y,grid_values) -> state index.
        for row in self.level.tiles:
            for tile in row:
                # Do not need states for blank/wall tiles since player cannot be on those
                if tile.tile_type not in [ti.LevelTileType.FLOOR, ti.LevelTileType.WATER, ti.LevelTileType.TARGET, ti.LevelTileType.HARD_ICE]:
                    continue

                # compute all grid tile options
                grid_options = self.level.get_all_possible_surrounding_grid_types(tile.position)

                for grid in grid_options:
                    key = tile.position + grid
                    # map key to state number
                    self._to_state[key] = state_num

                    # include this state index in the list of target states if the tile is a TARGET
                    if tile.tile_type == ti.LevelTileType.TARGET:
                        self._target.append(state_num)
                    state_num += 1

        self._to_cell = {v: k for k, v in self._to_state.items()}  # maps state # to (x, y)

        # Define number of actions and states
        self._n_actions = len(ti.PlayerActions)
        self._n_states = state_num

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)  #randomly select an action
        self.observation_space = spaces.Discrete(state_num)  #state space

        # Displaying the grid in a window: human
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 32
            self.window_size = 600
            #till we need it!
            self.window = None
            self.clock = None

        # Load tiles images
        self.tile_images = {}
        asset_path = "gymnasium_env/assets/"

        try:
            #map letters to images
            map_images = {
                'PF': 'floor_with_player.png',
                'PT': 'target_with_player.png',     
                'PW': 'water_with_player.png',
                'PH': 'hard_ice_with_player.png',
                'W': 'wall.webp',      
                'T': 'target.webp',        
                'F': 'floor.webp',  
                'B': 'blank.webp',
                'H': 'hard_ice.png',
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

    def reset(self, seed=None, options=None):
        # Reset environment, level, and visited tiles
        super().reset(seed=seed)
        self.level.reset()

        # Determine next state
        player_position = self.level.player_position
        player_tile = self.level.get_tile(player_position)
        surrounding_tile_types = self.level.get_surrounding_grid_types(player_position)
        # Set next state in observation
        state_rep = player_position + tuple(surrounding_tile_types)
        obs = self._to_state[state_rep]

        # Reset visited tiles to be just the current position
        self.visited_tiles = set()
        self.visited_tiles.add(player_position + (player_tile.tile_type.value,))

        # Debug information
        info = {} 

        if self.render_mode == "human": 
            # self.level.render() #for terminal debugging
            self.render_pygame()
        return obs, info


    def step(self, action):
        all_tiles_covered = False
        # Perform action
        target_reached = self.level.perform_action(ti.PlayerActions(action))

        # Determine next state
        player_position = self.level.player_position
        player_tile = self.level.get_tile(player_position)
        surrounding_tile_types = self.level.get_surrounding_grid_types(player_position)
        # Set next state in observation
        state_rep = player_position + tuple(surrounding_tile_types)
        obs = self._to_state[state_rep]

        terminated = target_reached
        reward = 0

        visited_info = player_position + (player_tile.tile_type.value,)
        if target_reached:
            all_tiles_covered =( (len(self.visited_tiles)+1) == self.level.n_visitable_tiles )
            if all_tiles_covered:
                reward += 25
            reward += 10

        elif player_tile.tile_type == ti.LevelTileType.WATER:
            # If player lands on water tile, end episode and give deinfluencing reward
            reward = -1
            terminated = True
        elif (visited_info not in self.visited_tiles):
            # If player is visiting a new tile, give a reward of 1
            reward += 1
            self.visited_tiles.add(visited_info)
        
        # Debug information
        info = {
            'all_tiles_covered':all_tiles_covered,
            'target_reached': target_reached
        }

        if self.render_mode == "human":
            print(f"Action taken: {ti.PlayerActions(action)}")
            self.render_pygame() 
        
        # Return observation, reward, done, truncated (not used [eg: after 200 steps stop]), info
        return obs, reward, terminated, False, info
    

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
                    else:
                        print("Warning: No image for tile type", tile_char)
                else:
                    print("Warning: No tile at position", (j, i))

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
            elif current_tile.tile_type == ti.LevelTileType.HARD_ICE:
                peng_img = self.tile_images.get('PH')
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
    
    def get_available_actions_mask(self, state) -> int:
        state_rep = self._to_cell[state]
        tile = self.level.get_tile((state_rep[0], state_rep[1]))
        return self.level.get_available_actions(tile)

    def get_termination_actions(self, state) -> list[int]:
        state_rep = self._to_cell[state]
        tile = self.level.get_tile((state_rep[0], state_rep[1]))
        return self.level.get_termination_actions(tile.position)
    
    @staticmethod
    def get_all_possible_masks(available_action_mask: int) -> list[int]:
        possible_masks = []
        sub = available_action_mask
        while True:
            possible_masks.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & available_action_mask
        return possible_masks
    
    @staticmethod
    def action_mask_to_actions(action_mask) -> list[int]:
        available_actions = []
        for action in ti.PlayerActions:
            if ((action_mask >> (len(ti.PlayerActions) -1 - action.value)) & 1):
                available_actions.append(action.value)
        return available_actions
    
    def get_actions_boolean_list(self,available_actions):
        boolean_mask = [False] * self.n_actions
        
        for action in available_actions:
            if 0 <= action < self.n_actions: 
                boolean_mask[action] = True
        
        return boolean_mask


# Run to test the environment
if __name__ == "__main__":
    env = gym.make('thin-ice-v1', render_mode='human', level_str='level_13.txt')

    print("======================================== Check environment begin ========================================")
    check_env(env, warn=True)
    print("========================================  Check environment end  ========================================")

    print(env.unwrapped.level)
    # Reset the environment
    obs, info = env.reset()

    print("======================================== START ROUND ========================================")
    # Take a random action
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Game over!")
            break
