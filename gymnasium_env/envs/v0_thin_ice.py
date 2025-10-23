from enum import Enum
import numpy as np
from typing import Final

PATH_TO_LEVELS: Final[str] = './level_txt_files/'

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
    
    def reset(self):
        self._player_position = self.player_start

        #  Reset tiles
        for row in self.tiles:
            for tile in row:
                if tile.tile_type == LevelTileType.WATER:
                    tile.tile_type = LevelTileType.FLOOR

    def generate_tiles(self, level_str: str):
        tiles_list = []
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
                        case _:
                            tile_type = None
                    
                    if tile_type is not None:
                        row.append(Tile(tile_type, tile_position))
                tiles_list.append(row)
        return np.array(tiles_list, dtype=object)
    
    # Called in step function, returns True if reached target
    def perform_action(self, action: PlayerActions) -> bool:         
        new_pos = (self.player_position[0] + action.get_direction()[0], self.player_position[1] + action.get_direction()[1])

        # Check if the new position is valid
        new_tile = self.get_tile(new_pos)
        old_tile = self.get_tile(self.player_position)
        if new_tile is not None and new_tile.tile_type != LevelTileType.WALL:
            if old_tile.tile_type == LevelTileType.FLOOR:
                old_tile.tile_type = LevelTileType.WATER

            print("Moving to ", new_pos)
            self._player_position = new_pos
        else:
            print("Invalid move to ", new_pos)
        
        return np.array_equal(self.player_position, self.target)
    
    def get_available_actions(self):
        available_actions = []
        for action in PlayerActions:
            new_pos = (self.player_position[0] + action.get_direction()[0], self.player_position[1] + action.get_direction()[1])
            new_tile = self.get_tile(new_pos)
            if new_tile is not None and new_tile.tile_type not in(LevelTileType.WALL, LevelTileType.WATER):
                available_actions.append(action.value)
        if len(available_actions) == 0:
            # No available actions, add all of them, player will lose anyways
            available_actions.append(action in PlayerActions)
        
        return available_actions
    
    def render(self):
        print(self)

    # Gets the tile at a given position, or None if out of bounds
    def get_tile(self, position: tuple) -> Tile:
        if position[0] < 0 or position[1] < 0 or position[0] >= self.n_cols or position[1] >= self.n_rows:
            return None
        return self.tiles[position[1]][position[0]]
    
    @property
    def n_rows(self):
        return len(self.tiles)
    
    @property
    def n_cols(self):
        return len(self.tiles[0]) if len(self.tiles) > 0 else 0
    
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

