from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from ..objects import EnvLockedDoor, COLORS, Lever
from typing import List
import numpy as np

class ColorOrderedLeversGame(BasePuzzleGame):
    """
    Generated puzzle game room for the two pressure plate game.
    
    In this game, the door is only unlocked when the two players
    are simultaneously on both pressure plates at once. These pressure
    plates are constrained to ensure an agent standing on a plate cannot
    see the other plate.
    """

    mission = 'toggle the levers in the correct order'
    color_states = [
        'orange',
        'green',
        'cyan',
        'purple',
        'pink',
        'white',
    ]

    def __init__(self, width: int, height: int, env, exit_walls: List[WALL_SIDE], config: dict):
        self.num_levers = 2
        self.disable_length = 10
        self.disable_step = 0
        self.ordered_counter = -1
        # Call the super class
        super().__init__(width, height, env, exit_walls, config)

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=self.np_random.choice(list(COLORS)), state=EnvLockedDoor.states.locked)

        # Generate the ordered levers
        self.levers = []
        for i in range(self.num_levers):
            self.levers.append(Lever(color=self.color_states[i]))


        # Generate position and try to place the levers
        for lever in self.levers:
            loc = self._gen_loc()
            # Make sure we don't block the exit!
            while np.linalg.norm(np.array(loc) - np.array(exit)) <= 1.0:
                loc = self._gen_loc()
            self._set(*loc, lever)

        # store the objects
        self._set(*exit, self.exit_door)
        self.exits = [self.exit_door]

    def update(self):
        # If we're disabled, just skip the loop.
        if self.env.step_count <= self.disable_step:
            if self.env.step_count == self.disable_step:
                for lever in self.levers:
                    lever.enable()
            return 0, {}

        # Check the states of the levers, and unlock the door if relevant
        rew = np.zeros((len(self.env.agents, )), dtype=float)

        # Start tracking orders!
        ordered_counter = -1
        disable = False
        for i,lever in enumerate(self.levers):
            if lever.state == Lever.states.on:
                if ordered_counter == i-1:
                    ordered_counter = i
                else:
                    # This is on out of order!
                    disable = True
        # Also compare to the last order
        if self.ordered_counter == ordered_counter \
            or self.ordered_counter == ordered_counter - 1:
            self.ordered_counter = ordered_counter
            # reward the relevant agent
            if ordered_counter != -1:
                lever = self.levers[ordered_counter]
                if lever.last_toggling_agent is not None:
                    i = self.env.agents.index(
                        lever.last_toggling_agent)
                    rew[i] += lever.reward
                    lever.reward = 0
                    lever.last_toggling_agent = None
        else:
            # This is out of order!
            disable = True
        
        # Now do what we need to do if we need to disable all levers
        if disable:
            for lever in self.levers:
                lever.disable()
            self.ordered_counter = -1
            self.disable_step = self.disable_length + self.env.step_count
            # Penalty for out of order behavior
            rew -= 1.0/self.num_levers/len(self.env.agents)

        # Now what we need to do if we have all things done
        if self.ordered_counter == self.num_levers - 1 \
            and self.exit_door.state == EnvLockedDoor.states.locked:
            for lever in self.levers:
                lever.freeze()
            self.exit_door.unlock()
            rew += 1
        
        return rew, {}

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class ColorOrderedLeversGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = ColorOrderedLeversGame
    config = {}
    name = "ColorOrderedLevers"

