from .basegenerator import WALL_SIDE, BasePuzzleGame, BasePuzzleGameGenerator
from typing import List

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
        pass

    def update(self):
        # Check the states of the two pressure plates, and unlock the door
        # if the two plates are activated at once, giving a small reward.
        pass

    #def gen_room_obs(self):
        # IDK what I'd do with this yet

class TwoPressurePlateGameGenerator(BasePuzzleGameGenerator):

    # Define the branching factor, in case we make multiple path rooms.
    BRANCHES: int = 1
    # Declare that we do need to check update for this room.
    NEEDS_UPDATE: bool = True
    
    PuzzleGame = TwoPressurePlateGame
    config = {}

