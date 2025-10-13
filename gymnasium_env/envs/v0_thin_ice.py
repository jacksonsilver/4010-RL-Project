from enum import Enum

# The actions that the player is capable of doing
class PlayerActions(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

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
        self.tile_type = tile_type
        self.position = position
    
    def __str__(self):
        return str(self.tile_type)
    
class Level:
    def __init__(self, path_to_level):
        self.tiles = [] # 2D array of arrays of tiles
        self.player_start = None # The starting position of the player
        self.target = None # The position of the target tile

        with open(path_to_level, 'r') as f:
            for line in f.readlines():
                row = [] # An array of tiles
                for char in line.strip():
                    if char == 'F':
                        row.append(Tile(LevelTileType.FLOOR, (len(row), len(self.tiles))))
                    elif char == 'W':
                        row.append(Tile(LevelTileType.WALL, (len(row), len(self.tiles))))
                    elif char == 'T':
                        self.target = (len(row), len(self.tiles))
                        row.append(Tile(LevelTileType.TARGET, (len(row), len(self.tiles))))
                    elif char == '~':
                        row.append(Tile(LevelTileType.WATER, (len(row), len(self.tiles))))
                    elif char == 'P':
                        self.player_start = (len(row), len(self.tiles))
                        row.append(Tile(LevelTileType.FLOOR, (len(row), len(self.tiles))))
                    elif char == 'B':
                        row.append(Tile(LevelTileType.BLANK, (len(row), len(self.tiles))))
                    
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
        match(action):
            case PlayerActions.LEFT:
                new_pos = (self.player_pos[0] - 1, self.player_pos[1] )
            case PlayerActions.RIGHT:
                new_pos = (self.player_pos[0] + 1, self.player_pos[1])
            case PlayerActions.UP:
                new_pos = (self.player_pos[0], self.player_pos[1] - 1)
            case PlayerActions.DOWN:
                new_pos = (self.player_pos[0], self.player_pos[1] + 1)
            case _:
                new_pos = self.player_pos
        
        # Check if the new position is valid
        tile = self.level.get_tile(new_pos)
        if tile is not None and tile.tile_type != LevelTileType.WALL:
            print("Moving to ", new_pos)
            self.player_pos = new_pos
        else:
            print("Invalid move to ", new_pos)
        
        return self.player_pos == self.target_pos

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
