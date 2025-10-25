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
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None, level_str='level_0.txt'):
        self.render_mode = render_mode

        # Initialize the level
        self._level = ti.Level(level_str)

        self._to_state = {}  # maps (x, y, W-MASK, AVAIL-ACTION-MASK) to state #

        # collect all possible target state indices (any tile that is a TARGET)
        self._target = []

        state_num = 0
        # Build mapping from (x,y,water_mask,avail_mask) -> state index.
        for row in self.level.tiles:
            for tile in row:
                # Do not need states for blank/wall tiles since player cannot be on those
                if tile.tile_type not in [ti.LevelTileType.FLOOR, ti.LevelTileType.WATER, ti.LevelTileType.TARGET]:
                    continue

                # water_options determines the possible values in the state representation for the water mask
                water_options = []
                if tile.tile_type == ti.LevelTileType.FLOOR:
                    water_options = [ti.LevelTileType.FLOOR.get_water_mask(), ti.LevelTileType.WATER.get_water_mask()]
                else:
                    water_options = [tile.tile_type.get_water_mask()]

                # Compute available actions mask for this tile (4-bit integer)
                avail_mask = self._level.get_available_actions(tile)

                # for each water option and for each submask of avail_mask, create a state
                # For example, if the avail_mask was 1010, this would create submasks of 1010, 0010, 1000, and 0000.
                for wmask in water_options:
                    sub = avail_mask
                    while True:
                        key = tile.position + (wmask, sub)
                        # map key to state number
                        self._to_state[key] = state_num

                        # include this state index in the list of target states if the tile is a TARGET
                        if tile.tile_type == ti.LevelTileType.TARGET:
                            self._target.append(state_num)
                        
                        state_num += 1
                        if sub == 0:
                            break
                        sub = (sub - 1) & avail_mask

        self._to_cell = {v: k for k, v in self._to_state.items()}  # maps state # to (x, y)

        # Define number of actions and states
        self._n_actions = len(ti.PlayerActions)
        self._n_states = state_num

        # Set visited tiles parameter
        self.visited_tiles = set()

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)  #randomly select an action
        # Observations are scalar state indices in [0, n_states-1]
        self.observation_space = spaces.Discrete(self._n_states)

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

    def reset(self, seed=None, options=None):
        # Reset environment, level, and visited tiles
        super().reset(seed=seed)
        self.level.reset()
        self.visited_tiles = set()

        # Determine initial state
        player_position = self.level.player_position
        player_tile = self.level.get_tile(player_position)
        water_mask = player_tile.tile_type.get_water_mask()
        avail_mask = self.level.get_available_actions(player_tile)
        # Set next state in observation
        obs = self._to_state[player_position + (water_mask, avail_mask)]

        # Debug information
        info = {} 

        if self.render_mode == "human": 
            # self.level.render() #for terminal debugging
            self.render_pygame()
        return obs, info

    def step(self, action):
        # Perform action
        target_reached, position_changed = self.level.perform_action(ti.PlayerActions(action))

        # Determine reward
        self.visited_tiles.add(self.level.player_position)
        reward = len(self.visited_tiles) / self.level.n_visitable_tiles if position_changed else 0
        reward = reward + 1 if target_reached else reward

        # Set terminated if target is reached
        terminated = target_reached

        # Determine next state
        player_position = self.level.player_position
        player_tile = self.level.get_tile(player_position)
        water_mask = player_tile.tile_type.get_water_mask()
        avail_mask = self.level.get_available_actions(player_tile)
        # Set next state in observation
        obs = self._to_state[player_position + (water_mask, avail_mask)]

        # If player lands on water tile, end episode and give deinfluencing reward
        if player_tile.tile_type == ti.LevelTileType.WATER:
            reward = 0
            terminated = True
        
        # Debug information
        info = {}

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
            print("Game over!")
            break
        