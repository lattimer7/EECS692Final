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
    For the first time a correct color is selected, a reward of 0.5 is given to all and doubled
    for the toggler, and when all boxes are done, a reward of 1 is given. Finally, another
    reward of 1 is given when the agents exit the room.

    Only one of the two agents can see the triangles, while only the other can cycle the box color.
    """

    mission = 'match the color of the boxes to their corresponding triangles'

    def __init__(self, width: int, height: int, env, exit_walls: List[WALL_SIDE], config: dict):
        self.code_size = 4
        # Call the super class
        super().__init__(width, height, env, exit_walls, config)
        self.most_right = 0

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=random.choice(list(COLORS)), state=EnvLockedDoor.states.closed)

        # Generate 4 directional sets
        dirs = np.random.randint(4, size=self.code_size)
        # Generate 4 colors
        colors = [random.choice(list(COLORS)) for _ in range(self.code_size)]
        # Select 1 agent to do the toggling and set obfuscation
        for agent in self.env.agents:
            agent.hide_item_types = [ColorCyclerBox().type]
        choice_agent = random.choice(self.env.agents)
        choice_agent.hide_item_types = [StaticCodedTriangle().type]


        # The center will always be the coded triangle
        # RIGHT DOWN LEFT UP
        def loc_gen(dir):
            loc1 = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
            if dir % 2 == 0:
                loc2 = (loc1[0] - (dir - 1), loc1[1])
            else:
                loc2 = (loc1[0], loc1[1] + (dir - 2))
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
                (StaticCodedTriangle(color=color, dir=dir),
                ColorCyclerBox(color=color2, interactable_agents=[choice_agent]),
                [0,0])
            )
            # Get a location
            loc = loc_gen(dir)
            while not loc_valid(loc): loc = loc_gen(dir)
            # Place it
            self.objs[loc[0]] = self.codepairs[-1][0]
            self.objs[loc[1]] = self.codepairs[-1][1]
            # and save another reference to the triangle location
            self.codepairs[-1][2][0] = loc[0][0]
            self.codepairs[-1][2][1] = loc[0][1]

        # save the exit
        self.objs[exit] = self.exit_door
        self.exits = [exit]

    def update(self):
        # Check the states of the codepairs and give a reward if the right
        # color is selected the first time.
        rew = np.zeros((len(self.env.agents, )), dtype=float)

        # TODO add info for toggle colors

        # Give reward for (first-time) correct box toggles
        num_right = 0
        for code,box,loc in self.codepairs:
            if code.color == box.color:
                if box.reward > 0:
                    # i = next(i for i,a in enumerate(self.env.agents) if a == box.last_toggling_agent)
                    # rew[i] += box.reward
                    # Also give overall reward
                    for i,a in enumerate(self.env.agents):
                        # Toggling agent gets double reward
                        if a == box.last_toggling_agent:
                            rew[i] += box.reward
                        # all other agents in view of code will get the reward
                        if a.in_view(*loc):
                            rew[i] += box.reward
                    box.reward = 0
                num_right += 1
        
        # give everyone a reward for sequences
        rew += max((num_right-self.most_right-1, 0))
        self.most_right = max(num_right, self.most_right)

        # If all the boxes are right, give another reward,
        # unlock the door, and disable all toggles
        if num_right == self.code_size and self.exit_door.state == EnvLockedDoor.states.closed:
            rew += 1
            self.exit_door.unlock()
            for _,box,_ in self.codepairs:
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

