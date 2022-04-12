from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from ..objects import EnvLockedDoor, COLORS, StaticCodedTriangle, ColorCyclerBox
from typing import List
import numpy as np
import random

class CycleBoxColorGame(BasePuzzleGame):
    """
    Generated puzzle game room for the two pressure plate game.
    
    In this game, the door is only unlocked when the boxes match the colors of
    the triangles. The two are spawned next to each other in randomized locations
    around the room (though a set of trianglecode and cyclebox are never spawned next to another).
    For the first time a correct color is selected, a reward of 0.5 is given, and when all boxes
    are done, a reward of 1 is given. Finally, another reward of 1 is given when the agents exit the room.

    Only one of the two agents can see the triangles, while only the other can cycle the box color.
    """

    mission = 'match the color of the boxes to their corresponding triangles'

    def __init__(self, width: int, height: int, exit_walls: List[WALL_SIDE], config: dict):
        # Call the super class
        super().__init__(width, height, exit_walls, config)
        self.code_size = 4

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=random.choice(list(COLORS)), state=EnvLockedDoor.states.closed)

        # Generate 4 directional sets
        dirs = np.random.randint(self.code_size, size=self.code_size)
        # Generate 4 colors
        colors = [random.choice(list(COLORS)) for _ in range(self.code_size)]
        # Select 1 agent to do the toggling and set obfuscation
        for agent in self.agents:
            agent.hide_item_types = [ColorCyclerBox]
        choice_agent = random.choice(self.agents)
        choice_agent.hide_item_types = [StaticCodedTriangle]


        # The center will always be the coded triangle
        def loc_gen(dir):
            loc1 = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
            if dir < 2:
                loc2 = (loc1[0] + dir*2 - 1, loc1[1])
            else:
                loc2 = (loc1[0], loc1[1] + (dir-2)*2 - 1)
            return (loc1,loc2)

        def loc_valid(loc_pair):
            if loc_pair[1][0] == 0 or loc_pair[1][0] == self.width - 1 \
                or loc_pair[1][1] == 0 or loc_pair[1][1] == self.height - 1:
                return False
            if self._is_free(*loc_pair[0]) and self._is_free(*loc_pair[1]):
                return True
            return False
        
        def rng_exclude_color(color):
            ret = random.choice(list(COLORS))
            while color == ret: ret = random.choice(list(COLORS))
            return ret

        # Create the objects (4 colors) and save them
        self.codepairs = []
        for color, dir in zip(colors, dirs):
            color2 = rng_exclude_color(color)
            self.codepairs.append(
                (StaticCodedTriangle(color),
                ColorCyclerBox(color=color2, interactable_agents=[choice_agent]))
            )
            # Get a location
            loc = loc_gen(dir)
            while not loc_valid(loc): loc = loc_gen(dir)
            # Place it
            self.objs[loc[0]] = self.codepairs[-1][0]
            self.objs[loc[1]] = self.codepairs[-1][1]

        # save the exit
        self.exits = [exit]

    def update(self):
        # Check the states of the codepairs and give a reward if the right
        # color is selected the first time.
        rew = np.zeros((len(self.agents, )), dtype=float)

        # TODO add info for toggle colors

        # Give reward for (first-time) correct box toggles
        all_right = True
        for code,box in self.codepairs:
            if code.color == box.color:
                if box.reward > 0:
                    i = next(i for i,a in enumerate(self.agents) if a == box.last_toggling_agent)
                    rew[i] += box.reward
                    box.reward = 0
            else:
                all_right = False
        
        # If all the boxes are right, give another reward,
        # unlock the door, and disable all toggles
        if all_right:
            rew += 1
            self.exit_door.unlock()
            for _,box in self.codepairs:
                box.disable()
        
        return rew, {}

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class CycleBoxColorGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = CycleBoxColorGame
    config = {}

