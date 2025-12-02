from enum import Enum
import numpy as np
from typing import Final

PATH_TO_LEVELS: Final[str] = './level_txt_files/'

def set_bit(number, position):
  """
  Sets the bit at the given position in a number to 1.

  Args:
    number: The integer in which to set the bit.
    position: The 0-indexed position of the bit to set.

  Returns:
    The new number with the bit at the specified position set to 1.
  """
  mask = 1 << position
  return number | mask

# The actions that the player is capable of doing
class PlayerActions(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    def get_direction(self):
        match(self):
            case PlayerActions.LEFT:
                return (-1, 0)
            case PlayerActions.RIGHT:
                return (1, 0)
            case PlayerActions.UP:
                return (0, -1)
            case PlayerActions.DOWN:
                return (0, 1)

# The different types of tiles that can be in a level
class LevelTileType(Enum):
    BLANK = 0
    FLOOR = 1
    WATER = 2
    WALL = 3
    TARGET = 4
    HARD_ICE = 5

    def get_water_mask(self):
        match(self):
            case LevelTileType.WATER:
                return 1
            case _:
                return 0
    
    # Gets all types the tile can possibly be transformed into
    def get_possible_types(self):
        match(self):
            case LevelTileType.FLOOR:
                return [LevelTileType.FLOOR.value, LevelTileType.WATER.value]
            case LevelTileType.HARD_ICE:
                return [LevelTileType.HARD_ICE.value, LevelTileType.FLOOR.value, LevelTileType.WATER.value]
            case _:
                return [self.value]

    # For printing the first letter of the tile
    def __str__(self):
        match(self):
            case LevelTileType.BLANK:
                return 'B'
            case LevelTileType.FLOOR:
                return 'F'
            case LevelTileType.WATER:
                return '~'
            case LevelTileType.WALL:
                return 'W'
            case LevelTileType.TARGET:
                return 'T'
            case LevelTileType.HARD_ICE:
                return 'H'
            case _:
                return '?'

class Tile():
    def __init__(self, tile_type: LevelTileType, position: tuple):
        self._tile_type = tile_type
        self._init_tile_type = tile_type
        self._position = position
    
    def reset(self):
        self.tile_type = self._init_tile_type
    
    @property
    def tile_type(self):
        return self._tile_type
    
    @property
    def init_tile_type(self):
        return self._init_tile_type
    
    @tile_type.setter
    def tile_type(self, new_type: LevelTileType):
        self._tile_type = new_type
    
    @property
    def position(self):
        return self._position
    
    def __str__(self):
        return str(self.tile_type)
    
class Level:
    def __init__(self, level_str: str):
        self._tiles = self.generate_tiles(level_str)
        self._player_position = self.player_start
        self._n_visitable_tiles = self.get_visitable_tile_count()
    
    def reset(self):
        self._player_position = self.player_start

        #  Reset tiles
        for row in self.tiles:
            for tile in row:
                tile.reset()

    def generate_tiles(self, level_str: str):
        tiles_list = []
        self._n_tile_types = 6 #make sure to change
        with open(PATH_TO_LEVELS + level_str, 'r') as f:
            for line in f.readlines():
                row = [] # An array of tiles
                for char in line.strip():
                    tile_position = (len(row), len(tiles_list)) # X,Y position
                    tile_type: LevelTileType = None
                    match char:
                        case 'F':
                            tile_type = LevelTileType.FLOOR
                        case 'W':
                            tile_type = LevelTileType.WALL
                        case 'T':
                            tile_type = LevelTileType.TARGET
                            self._target = tile_position
                        case '~':
                            tile_type = LevelTileType.WATER
                        case 'P':
                            tile_type = LevelTileType.FLOOR
                            self._player_start = tile_position
                        case 'B':
                            tile_type = LevelTileType.BLANK
                        case 'H':
                            tile_type = LevelTileType.HARD_ICE
                        case _:
                            tile_type = None
                    
                    if tile_type is not None:
                        row.append(Tile(tile_type, tile_position))
                tiles_list.append(row)
        return np.array(tiles_list, dtype=object)
    

    def get_visitable_tile_count(self):
        count = 0
        for row in self.tiles:
            for tile in row:
                if tile.tile_type in [LevelTileType.FLOOR, LevelTileType.TARGET]:
                    count += 1
                elif tile.tile_type == LevelTileType.HARD_ICE:
                    count += 2
        return count

    def get_visited_tile_count(self):
        count = 0
        for row in self.tiles:
            for tile in row:
                # Any tile that is water means that it has been visited
                if tile.tile_type == LevelTileType.WATER:
                    count += 1
                
                # If a tile's initial type was hard ice and it it's current is not, it means it has been visited.
                # Previous check accounts for double visit
                if tile.init_tile_type == LevelTileType.HARD_ICE and tile.tile_type != tile.init_tile_type:
                    count += 1
        return count

    # Called in step function, returns True if reached target
    def perform_action(self, action: PlayerActions):         
        new_pos = (self.player_position[0] + action.get_direction()[0], self.player_position[1] + action.get_direction()[1])

        # Check if the new position is valid
        new_tile = self.get_tile(new_pos)
        old_tile = self.get_tile(self.player_position)
        if new_tile is not None and new_tile.tile_type != LevelTileType.WALL:
            # If player walks off hard ice, it becomes floor
            if old_tile.tile_type == LevelTileType.HARD_ICE:
                old_tile.tile_type = LevelTileType.FLOOR
            
            # If player walks off floor, it becomes water
            elif old_tile.tile_type == LevelTileType.FLOOR:
                old_tile.tile_type = LevelTileType.WATER
            
            #print("Moving to ", new_pos)
            self._player_position = new_pos
        else:
            print("Invalid move to ", new_pos)
        
        # Return if target reached and if player successfully moved
        return np.array_equal(self.player_position, self.target)
    
    def get_available_actions(self, tile: Tile) -> int:
        # Return a 4-bit mask (as int) where bit i corresponds to PlayerActions with value i
        mask = 0
        for action in PlayerActions:
            dx, dy = action.get_direction()
            new_pos = (tile.position[0] + dx, tile.position[1] + dy)
            new_tile = self.get_tile(new_pos)

            # If action is available (not out of bounds and not wall or water), set the corresponding bit
            if new_tile is not None and new_tile.tile_type not in (LevelTileType.WALL, LevelTileType.WATER):
                mask = set_bit(mask, len(PlayerActions) - 1 - action.value)

        return mask
    
    def render(self):
        print(self)

    # Gets the tile at a given position, or None if out of bounds
    def get_tile(self, position: tuple) -> Tile:
        if position[0] < 0 or position[1] < 0 or position[0] >= self.n_cols or position[1] >= self.n_rows:
            return None
        return self.tiles[position[1]][position[0]]

    def get_surrounding_grid_types(self, position: tuple) -> list[int]:
        # Gets the tile types (in ints) of the tiles in a 3x3 grid centered at the given position
        tile = self.get_tile(position)
        if tile is None:
            return []
        
        types = []
        # Loop through the 3x3 grid, starting at top left corner, with bottom right corner last
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                neighbor_pos = (position[0] + dx, position[1] + dy)
                neighbor_tile = self.get_tile(neighbor_pos)
                if neighbor_tile is not None:
                    types.append(neighbor_tile.tile_type.value)
                else:
                    types.append(LevelTileType.BLANK.value)
        return types
    
    def get_all_possible_surrounding_grid_types(self, position: tuple) -> list[tuple[int]]:
        # Gets all possible combinations of tile types (in ints) of the tiles in a 3x3 grid centered at the given position
        current_grid = self.get_surrounding_grid_types(position)
        if len(current_grid) == 0:
            return []
        
        # Go through each tile in the grid and get all possible types for that tile and add to list of list
        tile_types_options = []

        for i in range(len(current_grid)):
            i_tile_type = current_grid[i]
            i_all_options = LevelTileType(i_tile_type).get_possible_types()
            tile_types_options.append(i_all_options)
        
        # Get all possible combinations of tile types in tile_types_options list
        # I'm so sorry about this code I just genuinely couldn't figure out a better way to do it
        grid_options = set()
        for i, type1 in enumerate(tile_types_options[0]):
            for j, type2 in enumerate(tile_types_options[1]):
                for k, type3 in enumerate(tile_types_options[2]):
                    for l, type4 in enumerate(tile_types_options[3]):
                        for m, type5 in enumerate(tile_types_options[4]):
                            for n, type6 in enumerate(tile_types_options[5]):
                                for o, type7 in enumerate(tile_types_options[6]):
                                    for p, type8 in enumerate(tile_types_options[7]):
                                        for q, type9 in enumerate(tile_types_options[8]):
                                            grid_options.add((type1, type2, type3, type4, type5, type6, type7, type8, type9))
        return list(grid_options)
    
    def get_termination_actions(self, position: tuple) -> list[int]:
        termination_actions = []
        # Returns the actions that would lead to termination (touching a water tile)
        for action in PlayerActions:
            dx, dy = action.get_direction()
            new_pos = (position[0] + dx, position[1] + dy)
            new_tile = self.get_tile(new_pos)
            if new_tile is not None and new_tile.tile_type == LevelTileType.WATER:
                termination_actions.append(action.value)
        return termination_actions
    
    @property
    def n_rows(self):
        return len(self.tiles)
    
    @property
    def n_cols(self):
        return len(self.tiles[0]) if len(self.tiles) > 0 else 0
    
    @property
    def n_visitable_tiles(self):
        return self._n_visitable_tiles
    
    @property
    def tiles(self):
        return self._tiles
    
    @property
    def player_position(self):
        return self._player_position
    
    @property
    def player_position_tile(self):
        return self.get_tile(self.player_position)
    
    @property
    def player_start(self):
        return self._player_start
    
    @property
    def target(self):
        return self._target
    
    @property
    def n_tile_types(self):
        return self._n_tile_types
    
    # For printing out the level
    def __str__(self):
        s = ''
        for i, row in enumerate(self.tiles):
            for j, tile in enumerate(row):
                if self.player_position == (i, j):
                    s += 'P'
                else:
                    s += str(tile)
            s += '\n'
        return s

