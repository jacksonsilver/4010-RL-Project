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
                return np.array([-1, 0])
            case PlayerActions.RIGHT:
                return np.array([1, 0])
            case PlayerActions.UP:
                return np.array([0, -1])
            case PlayerActions.DOWN:
                return np.array([0, 1])

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
    def __init__(self, tile_type: LevelTileType, position: np.array):
        self.tile_type = tile_type
        self.position = position
    
    def __str__(self):
        return str(self.tile_type)
    
class Level:
    def __init__(self, level_str: str):
        self.tiles = [] # 2D array of arrays of tiles
        self.player_start = None # The starting position of the player
        self.target = None # The position of the target tile

        with open(PATH_TO_LEVELS + level_str, 'r') as f:
            for line in f.readlines():
                row = [] # An array of tiles
                for char in line.strip():
                    tile_position = np.array([len(row), len(self.tiles)])
                    tile_type: LevelTileType = None
                    match char:
                        case 'F':
                            tile_type = LevelTileType.FLOOR
                        case 'W':
                            tile_type = LevelTileType.WALL
                        case 'T':
                            tile_type = LevelTileType.TARGET
                            self.target = tile_position
                        case '~':
                            tile_type = LevelTileType.WATER
                        case 'P':
                            tile_type = LevelTileType.FLOOR
                            self.player_start = tile_position
                        case 'B':
                            tile_type = LevelTileType.BLANK
                        case _:
                            tile_type = None
                    
                    if tile_type is not None:
                        row.append(Tile(tile_type, tile_position))
                self.tiles.append(row)

    # Gets the tile at a given position, or None if out of bounds
    def get_tile(self, position: tuple) -> Tile:
        if position[0] < 0 or position[1] < 0 or position[0] >= self.get_num_cols() or position[1] >= self.get_num_rows():
            return None
        return self.tiles[position[1]][position[0]]
    
    def get_num_rows(self):
        return len(self.tiles)

    def get_num_cols(self):
        return len(self.tiles[0]) if len(self.tiles) > 0 else 0
    
    # For printing out the level
    def __str__(self):
        s = ''
        for i, row in enumerate(self.tiles):
            for j, tile in enumerate(row):
                if self.player_start == (i, j):
                    s += 'P'
                else:
                    s += str(tile)
            s += '\n'
        return s

# The player to complete the Thin Ice environment
class ThinIcePlayer:
    def __init__(self, level: Level):
        self.level = level
        self.player_pos = level.player_start
        self.target_pos = level.target
    
    def reset(self, seed=None):
        self.player_pos = self.level.player_start
        self.target_pos = self.level.target
    
    # Called in step function, returns True if reached target
    def perform_action(self, action: PlayerActions) -> bool:         
        new_pos = self.player_pos + action.get_direction()

        # Check if the new position is valid
        tile = self.level.get_tile(new_pos)
        if tile is not None and tile.tile_type != LevelTileType.WALL:
            print("Moving to ", new_pos)
            self.player_pos = new_pos
        else:
            print("Invalid move to ", new_pos)
        
        return np.array_equal(self.player_pos, self.target_pos)

    def render(self):
        for i in range(self.level.get_num_rows()):
            for j in range(self.level.get_num_cols()):
                if self.player_pos == (j, i):
                    print('P', end='')
                else:
                    tile = self.level.get_tile((j, i))
                    if tile is not None:
                        print(str(tile), end='')
                    else:
                        print(' ', end='')
            print()
        print()
