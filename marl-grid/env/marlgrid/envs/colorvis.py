import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import Goal, Wall
from gym_minigrid.minigrid import Door, Key


def dis_func(x, y, k=1):
    return np.linalg.norm(x - y) / k

class ColorBlindMultiGrid(MultiGridEnv):
    """
    Environment with two keys and two doors. Must be run with selective grid agents
    """
    mission = 'unlock both doors and leave'
    metadata = {}

    def __init__(self, config):
        self.size = config.get('grid_size')
        width = self.size
        height = self.size

        super(ColorBlindMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height):
        """Generate grid without agents."""

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        self.red_door = Door(color='red', is_locked=True)
        self.blue_door = Door(color='blue', is_locked=True)
        self.red_key = Key(color='red')
        self.blue_key = Key(color='blue')
        
        doors = [self.red_door, self.blue_door]
        keys =  [self.red_key, self.blue_key]
        self.np_random.shuffle(doors)
        
        for color in keys:
            keyx = self.np_random.randint(2, self.width - 2)
            keyy = self.np_random.randint(2, self.height - 2)
            self.grid.set(keyx, keyy, color)
            color.pos = np.asarray([keyx, keyy])

        # Add a red/blue door at a random position in the left wall
        pos = self.np_random.randint(1, self.size - 1)
        self.grid.set(0, pos, doors[0])
        doors[0].pos = np.asarray([0, pos])

        # Add a red/blue door at a random position in the right wall
        pos = self.np_random.randint(1, self.width - 1)
        self.grid.set(self.width - 1, pos, doors[1])
        doors[1].pos = np.asarray([self.width - 1, pos])

        return None

    def _reward(self):
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _door_pos_to_one_hot(self, pos):
        p = np.zeros((self.width + self.height,))
        p[int(pos[0])] = 1.
        p[int(self.width + pos[1])] = 1.
        return p

    def gen_global_obs(self):
        # concat door state and pos into a 1-D vector
        door_state = np.array([int(self.red_door.is_open),
                               int(self.blue_door.is_open)])
        door_obs = np.concatenate([
            door_state,
            self._door_pos_to_one_hot(self.red_door.pos),
            self._door_pos_to_one_hot(self.blue_door.pos)])
        obs = {
            'door_obs': door_obs,
            'comm_act': np.stack([a.comm for a in self.agents],
                                 axis=0),  # (N, comm_len)
            'env_act': np.stack([a.env_act for a in self.agents],
                                axis=0),  # (N, 1)
        }
        return obs

    def reset(self):
        obs_dict = MultiGridEnv.reset(self)
        obs_dict['global'] = self.gen_global_obs()
        return obs_dict

    def step(self, action_dict):
        #red_door_opened_before = self.red_door.is_open
        #blue_door_opened_before = self.blue_door.is_open

        obs_dict, _, _, info_dict = MultiGridEnv.step(self, action_dict)

        step_rewards = np.zeros((self.num_agents, ), dtype=float)

        red_door_opened = self.red_door.is_open
        blue_door_opened = self.blue_door.is_open


        done = False
        success = False
        if red_door_opened and blue_door_opened:
            success = True
            done = True


        timeout = (self.step_count >= self.max_steps)

        obs_dict['global'] = self.gen_global_obs()
        rew_dict = {f'agent_{i}': step_rewards[i] for i in range(
            len(step_rewards))}
        done_dict = {'__all__': done or timeout}
        info_dict = {
            'done': done,
            'timeout': timeout,
            'success': success,
            'comm': obs_dict['global']['comm_act'].tolist(),
            'env_act': obs_dict['global']['env_act'].tolist(),
            't': self.step_count,
            # 'red_door_opened_now': red_door_opened_now,
        }
        return obs_dict, rew_dict, done_dict, info_dict
