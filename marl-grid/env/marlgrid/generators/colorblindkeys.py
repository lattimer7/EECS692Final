from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from ..objects import EnvLockedDoor, COLORS, Key, KeyHole
from typing import List
import numpy as np

class ColorBlindGame(BasePuzzleGame):
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
    
    mission = 'unlock both doors and leave'

    def __init__(self, width: int, height: int, env, exit_walls: List[WALL_SIDE], config: dict):
        self.num_keys = 2
        # Call the super class
        super().__init__(width, height, env, exit_walls, config)
        self.most_right = 0
        self.viewing_agents = {}

    def _gen_objs(self):
        # Generate the locations of the two pressure plates & the location
        # of the exiting door & store to self.objs and self.exits.
        exit = self._sample_exit_walls()
        self.exit_door = EnvLockedDoor(color=self.np_random.choice(list(COLORS)), state=EnvLockedDoor.states.locked)

        # Select num_keys agents to force
        self.key_agents = self.np_random.choice(len(self.env.agents), size=self.num_keys, replace=False)
        
        # Generate the keys and keyholes
        self.keys = []
        self.keyholes = []
        for i in range(self.num_keys):
            # Get the relevant agent
            agent = self.env.agents[self.key_agents[i]]
            # Create the key
            key = Key(interactable_agents=[agent], color=agent.color)
            # Give the key a reward value
            key.reward = 0.5
            # Hide it from the relevant agent
            agent.hide_item_types.append(key.type)
            # Create the keyhole
            keyhole = KeyHole(key_obj=key, interactable_agents=[agent], color=agent.color)
            self.keys.append(key)
            self.keyholes.append(keyhole)

        # Place all the keys and keyholes into the world
        for keys in self.keys:
            loc = self._gen_loc()
            # Make sure we don't block the exit!
            # while np.linalg.norm(np.array(loc) - np.array(exit)) <= 1.0:
            #     loc = self._gen_loc()
            self._set(*loc, keys)

        # Seperate in case we want to prevent overlap on keyholes
        for keyholes in self.keyholes:
            loc = self._gen_loc()
            # Make sure we don't block the exit!
            # while np.linalg.norm(np.array(loc) - np.array(exit)) <= 1.0:
            #     loc = self._gen_loc()
            self._set(*loc, keyholes)

        # save the exit
        self._set(*exit, self.exit_door)
        self.exits = [self.exit_door]

    def prestep(self):
        # Get what agents can see the key *BEFORE* we update
        for key in self.keys:
            if key.reward > 0:
                self.viewing_agents[key] = []
                # Just add any agent that can see the key
                for i,a in enumerate(self.env.agents):
                    if key.type not in a.hide_item_types and a.in_view(*key.pos_init):
                        self.viewing_agents[key].append(i)
    
    def update(self):
        # Perform checks for unlocking doors and giving rewards
        rew = np.zeros((len(self.env.agents, )), dtype=float)

        # Check if any of the key agents are carrying a key
        # For any carrying agent, give all seeing agents a reward too
        for agent_id in self.key_agents:
            obj = self.env.agents[agent_id].carrying
            if obj is not None and obj in self.keys:
                if obj.reward > 0:
                    # Simplify code
                    key = obj
                    v_a = self.viewing_agents[key]
                    # Give carrying reward
                    rew[agent_id] += key.reward
                    # Give seeing reward
                    rew[v_a] += key.reward / 2
                    # Remove reward
                    key.reward = 0
        
        # give a reward to the unlocking and discovering agent
        num_unlocked = 0
        for keyhole in self.keyholes:
            if keyhole.unlocking_agent is not None:
                i = self.env.agents.index(keyhole.unlocking_agent)
                rew[i] += 1
                keyhole.unlocking_agent = None
            # don't give reward for discovery
            # if keyhole.last_toggling_agent is not None \
            #     and keyhole.reward > 0:
            #     i = self.env.agents.index(keyhole.last_toggling_agent)
            #     rew[i] += keyhole.reward
            #     keyhole.last_toggling_agent = None
            #     keyhole.reward = 0
            if keyhole.state == KeyHole.states.unlocked:
                num_unlocked += 1

        # If everything is unlocked, unlock the door
        if num_unlocked == self.num_keys and self.exit_door.state == EnvLockedDoor.states.locked:
            rew += 1
            self.exit_door.unlock()
        
        return rew, {}

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class ColorBlindGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = ColorBlindGame
    config = {}
    name = "ColorBlindKeys"

