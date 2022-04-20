import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import FreeDoor, PressurePlate, EnvLockedDoor


class PressurePlateMultiGrid(MultiGridEnv):
    """
    Single room with red and blue doors on opposite sides.
    The red door must be opened before the blue door to
    obtain a reward.
    """

    mission = 'open the red door then the blue door'

    def __init__(self, config):
        self.size = config.get('grid_size')
        width = self.size
        height = self.size

        super(PressurePlateMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height):
        """Generate grid without agents."""

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        self.red_door = EnvLockedDoor(0.5, color='red', state=FreeDoor.states.closed)

        self.pressure_plate = PressurePlate(0.5, color='orange')

        # Add a red/blue door at a random position in the left wall
        worh = self.np_random.randint(0,2)
        if worh == 0:
            pos = self.np_random.randint(1, self.size - 1)
            self.grid.set(0, pos, self.red_door)
            self.red_door.pos = np.asarray([0, pos])

            # Add a red/blue door at a random position in the right wall
        #    posy = self.np_random.randint(1, self.width - 1)
        #    posx = self.np_random.randint(self.width//2, self.width - 1)
        #    self.grid.set(posx, posy, self.pressure_plate)
        #    self.pressure_plate.pos = np.asarray([posx, posy])

        else:
        #    posy = self.np_random.randint(1, self.size - 1)
        #    posx = self.np_random.randint(1, self.size//2+1)
        #    self.grid.set(posx, posy, self.pressure_plate)
        #    self.pressure_plate.pos = np.asarray([posx, posy])

            # Add a red/blue door at a random position in the right wall
            pos = self.np_random.randint(1, self.width - 1)
            self.grid.set(self.width - 1, pos, self.red_door)
            self.red_door.pos = np.asarray([self.width - 1, pos])

        posy = self.np_random.randint(1, self.size - 1)
        posx = self.np_random.randint(1, self.width-1)
        self.grid.set(posx, posy, self.pressure_plate)
        self.pressure_plate.pos = np.asarray([posx, posy])
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
        door_state = np.array([int(self.red_door.is_open()),
                               int(any([agent.at_pos(self.pressure_plate.pos) for agent in self.agents]))])
        door_obs = np.concatenate([
            door_state,
            self._door_pos_to_one_hot(self.red_door.pos),
            self._door_pos_to_one_hot(self.pressure_plate.pos)])

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
        actions = [action_dict[f'agent_{i}'][0] for i in range(len(self.agents))]
        red_door_opened_before = self.red_door.been_toggled
        pressure_plate_active_before = any([agent.at_pos(self.pressure_plate.pos) for agent in self.agents])
        pressure_plate_agents_before =  np.array([agent.at_pos(self.pressure_plate.pos) for agent in self.agents])
        
        obs_dict, _, _, info_dict = MultiGridEnv.step(self, action_dict)

        #step_rewards = np.zeros((self.num_agents, ), dtype=float)
        step_rewards = np.array([float(self.pressure_plate.stepped()) if agent.at_pos(self.pressure_plate.pos) else float(0) for agent in self.agents], dtype=float)

        red_door_tried_after = self.red_door.been_toggled
        pressure_plate_active_after = any([agent.at_pos(self.pressure_plate.pos) for agent in self.agents])
        pressure_plate_agents_after =  np.array([agent.at_pos(self.pressure_plate.pos) for agent in self.agents])

        if (not red_door_opened_before) and red_door_tried_after:
            step_rewards = step_rewards + np.array([float(self.red_door.reward) if actions[i]==5 else float(0) for i, agent in enumerate(self.agents)], dtype=float)

        done = [self.agents[i].at_pos(self.red_door.pos) for i in range(self.num_agents)]
        
        if any(done):
            self.red_door.unlock()
            step_rewards += self._reward()
        elif pressure_plate_active_after:
            self.red_door.unlock()
        else:
            self.red_door.lock()

        #for i,d in enumerate(done):
        #    if d is False: continue
            # Give reward and deactivate the agent if done.
        #    if self.agents[i].done is False:
        #        self.agents[i].done = True
        #        step_rewards[i] += 1
        #    self.agents[i].deactivate()
        
        success = any(done)

        timeout = (self.step_count >= self.max_steps)

        obs_dict['global'] = self.gen_global_obs()
        rew_dict = {f'agent_{i}': step_rewards[i] for i in range(
            len(step_rewards))}
        done_dict = {'__all__': success or timeout}
        info_dict = {
            'done': success,
            'timeout': timeout,
            'success': success,
            'comm': obs_dict['global']['comm_act'].tolist(),
            'env_act': obs_dict['global']['env_act'].tolist(),
            't': self.step_count,
            'pressure_plate_active_now': self.red_door.is_open(),
        }
        return obs_dict, rew_dict, done_dict, info_dict
