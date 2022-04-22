from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from ..objects import EnvLockedDoor, COLORS, PressurePlate
from typing import List
import numpy as np
import random

class SinglePressurePlateGame(BasePuzzleGame):
    """
    Generated puzzle game room for the single pressure plate game.
    
    In this game, the door is only unlocked when one player is on a pressure
    plate the same time another player is attempting to open the door.
    """

    mission = 'activate a pressure plate and open the door simultaneously'

    def __init__(self, width: int, height: int, env, exit_walls: List[WALL_SIDE], config: dict):
        # Call the super class
        super().__init__(width, height, env, exit_walls, config)
        self.novel_door = True

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=random.choice(list(COLORS)),
                                       state=EnvLockedDoor.states.locked,
                                       require_open=True)

        self.pressureplate = PressurePlate()

        # For now just generate a random position
        loc = (self.np_random.randint(1, self.width - 1), self.np_random.randint(1, self.height - 1))

        # store the objects
        self.objs[exit] = self.exit_door
        self.objs[loc] = self.pressureplate
        self.exits = [exit]

    def update(self):
        # Check the states of the two pressure plates, and unlock the door
        # if the two plates are activated at once, giving a small reward.
        rew = np.zeros((len(self.env.agents, )), dtype=float)

        # Unlock the door if the plate is active
        if self.pressureplate.state == PressurePlate.states.active:
            self.exit_door.unlock()
        elif self.exit_door.state == EnvLockedDoor.states.closed:
            self.exit_door.lock()
            
        # Send the last toggle action again anyway
        if self.exit_door.last_toggling_agent is not None:
            # Do a discovery reward if novel
            if self.novel_door:
                i = self.env.agents.index(self.exit_door.last_toggling_agent)
                rew[i] += self.exit_door.reward
                self.novel_door = False
            # Send toggle signal
            self.exit_door.toggle(self.exit_door.last_toggling_agent, None)
            self.exit_door.last_toggling_agent = None
        
        # If the door is open, give rewards if just opened
        if self.exit_door.state == EnvLockedDoor.states.open \
            and self.exit_door.opening_agent is not None:
            # Reward the opening agent
            # i = self.env.agents.index(self.exit_door.opening_agent)
            # rew[i] += self.exit_door.reward
            # And reward both agents for opening
            self.exit_door.opening_agent = None
            rew += 1
        
        return rew, {}

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class SinglePressurePlateGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = SinglePressurePlateGame
    config = {}

