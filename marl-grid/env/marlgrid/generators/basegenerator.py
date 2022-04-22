from ..objects import Wall
from enum import IntEnum
from typing import List
import numpy as np

class WALL_SIDE(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    ALL = 4

 # X increasing to the right, Y increasing down.

class BasePuzzleGame:
    """
    Generated puzzle game room, where all objects are defined
    relative to the puzzle room origin (walls included). This class should
    store the state of a specific puzzle room, where the objects
    within that room are returned from get_objs() as a dict
    where the keys are tuples of (x_pos, y_pos) and the values
    are the objects. Inclusion of the boundary walls is unnecessary
    and will be ignored. get_objs() will also return a list of exits.

    gen_room_obs() will generate a dictionary of fix-sized
    room-specific observations of objects. This can be defined
    however you want, except that each dictionary entry must be
    a 1-D vector which will be concatenated with any matching entries.
    Generally, the keys should be "{object_name}_obs". Do not include
    objects such as walls.
    """

    mission = ''

    def __init__(self, width: int, height: int, env, exit_walls: List[WALL_SIDE], config: dict):
        self.width = width
        self.height = height
        self.env = env
        self.np_random = env.np_random
        self.exit_walls = exit_walls
        self.config = config
        self.objs = {}
        self.exits = []
        self._gen_objs()

    def get_objs(self):
        return self.objs, self.exits

    def update(self):
        return 0, {}

    def gen_room_obs(self):
        return {}

    def _gen_objs(self):
        pass

    def _set(self, x: int, y: int, obj):
        self.objs[(x, y)] = obj

    def _clear(self, x: int, y: int):
        return self.objs.pop((x, y), None)

    def _fill(self, x1: int, y1: int, x2: int, y2: int, obj_type=Wall):
        if obj_type is not None:
            for i in range(x1, x2):
                for j in range(y1, y2):
                    self._set(i, j, obj_type())
        else:
            for i in range(x1, x2):
                for j in range(y1, y2):
                    self._clear(i, j)

    def _is_free(self, x: int, y: int):
        return not ((x,y) in self.objs)

    def _sample_exit_walls(self):
        # get the size of the space to sample
        if len(self.exit_walls) == 1 and self.exit_walls[0] == WALL_SIDE.ALL:
            # treat this case specifically
            edge = self.np_random.randint(4)
        else:
            # sample the rest
            edge = self.np_random.randint(len(self.exit_walls))
            edge = int(self.exit_walls[edge])
        # This assumes the origin is the top left!
        if edge % 2 == 0:
            # left and right case
            loc = self.np_random.randint(1, self.height-1)
            return ((1 - edge//2)*(self.width-1), loc)
        else:
            # top and bottom case
            loc = self.np_random.randint(1, self.width-1)
            return (loc, (1 - edge//2)*(self.height-1))


class BasePuzzleGameGenerator:
    """
    Generator class to create the puzzle game objects. This class
    should define an object where the room sizes are stored along
    with any other parameters we want to use to generate the games.
    
    We define this as a seperate class so that we can use one generator
    to create multiple "instances" of the same game to lay out in
    the global world.
    """

    BRANCHES: int = 1
    NEEDS_UPDATE: bool = False
    
    PuzzleGame = BasePuzzleGame
    config = {}

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        # Merge the dictionaries if we have extra configs.
        if config is not None:
            self.config = {**self.config, **config}
    
    def generate(self, env, exit_walls: List[WALL_SIDE] = [WALL_SIDE.ALL]):
        return self.PuzzleGame(self.width, self.height, env, exit_walls, self.config)
