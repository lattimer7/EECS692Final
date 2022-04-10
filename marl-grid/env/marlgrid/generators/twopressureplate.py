from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from ..objects import EnvLockedDoor, COLORS, PressurePlate
from typing import List
import numpy as np
import random

class TwoPressurePlateGame(BasePuzzleGame):
    """
    Generated puzzle game room for the two pressure plate game.
    
    In this game, the door is only unlocked when the two players
    are simultaneously on both pressure plates at once. These pressure
    plates are constrained to ensure an agent standing on a plate cannot
    see the other plate.
    """

    mission = 'activate both pressure plates simultaneously'

    def __init__(self, width: int, height: int, exit_walls: List[WALL_SIDE], config: dict):
        # Call the super class
        super().__init__(width, height, exit_walls, config)

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=random.choice(list(COLORS)), state=EnvLockedDoor.states.closed)

        self.pressureplates = [
            PressurePlate(),
            PressurePlate()
        ]

        # For now just generate two random positions
        loc1 = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
        loc2 = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
        while loc2[0] == loc1[0] and loc2[1] == loc1[1]:
            loc2 = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
        
        # store the objects
        self.objs[exit] = self.exit_door
        self.objs[loc1] = self.pressureplates[0]
        self.objs[loc2] = self.pressureplates[1]
        self.exits = [exit]

    def update(self):
        # Check the states of the two pressure plates, and unlock the door
        # if the two plates are activated at once, giving a small reward.
        rew = 0
        if self.pressureplates[0].state == PressurePlate.states.active \
            and self.pressureplates[1].state == PressurePlate.states.active \
            and self.exit_door.state == EnvLockedDoor.states.closed:
            self.exit_door.unlock()
            rew = 1
        
        return rew, {}

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class TwoPressurePlateGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = TwoPressurePlateGame
    config = {}

