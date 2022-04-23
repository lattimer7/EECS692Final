import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import Wall
from ..generators import WALL_SIDE


class TwoRoomPuzzleMultiGrid(MultiGridEnv):
    """
    Two puzzle room environment, where the puzzles are
    selected from the provided generator classes. Each of these
    generators specifies how a room would be constructed as well
    as how the 

    with red and blue doors on opposite sides.
    The red door must be opened before the blue door to
    obtain a reward.
    """

    mission = 'open the red door then the blue door'

    def __init__(self, config):
        self.size = config.get('grid_size')
        # TODO: Change to be loaded by name from generators, for sake of
        self.generators = config.get('generators')

        self.intermediate_color = 'blue'
        self.exit_color = 'green'
        
        # These width's and height's are completely ignored down the line
        width = self.size
        height = self.size
        super(TwoRoomPuzzleMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height):
        """Generate grid without agents."""

        # Generate a random orientation and update the
        # overall width and height based on those values
        dir = self.np_random.randint(4)
        # RIGHT DOWN LEFT UP

        # Compute the width & height
        # Because we have an overlap of 1 multiply by
        # the size - 1 then add the extra 1 back
        unit = (self.size - 1)
        width = unit * (2 - dir % 2) + 1
        height = unit * (dir % 2 + 1) + 1
        self.width = width
        self.height = height
        self.max_dis = np.sqrt(np.sum(np.square([self.width, self.height])))

        # Compute the origin and exit cells (left, top)
        origin_cell = (unit * (dir == 2), unit * (dir == 3))
        exit_cell   = (unit * (dir == 0), unit * (dir == 1))

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(*origin_cell, self.size, self.size)
        self.grid.wall_rect(*exit_cell, self.size, self.size)

        # Sample a random generator
        gen_id = self.np_random.randint(len(self.generators), size=2)

        # Exit cell exit options
        entrance_side = (dir-2)%4
        exit_sides = list(range(4))
        exit_sides.remove(entrance_side)
        # Then valid exits
        exit_walls = [[WALL_SIDE(dir)],
                      [WALL_SIDE(d) for d in exit_sides]]
        
        # Get the generator objects and update mission
        # TODO: will this update the mission?
        self.games = [self.generators[v].generate(self, exit_walls[i]) for i,v in enumerate(gen_id)]
        self.mission = "TwoRoom: " + \
                        self.games[0].mission + "; " + \
                        self.games[1].mission

        # Add all objects that are not boundary walls.
        def place_obj(grid, origin, game):
            objs, exits = game.get_objs()
            for pos, obj in objs.items():
                # TODO: Will this skip all boundary wall types?
                if isinstance(obj, Wall):
                    if pos[0] != 0            \
                        and pos[1] != 0       \
                        and pos[0] != self.size-1 \
                        and pos[1] != self.size-1:
                            continue
                # Otherwise place the object into the grid.
                new_pos = (origin[0] + pos[0], origin[1] + pos[1])
                grid.set(*new_pos, obj)
                obj.pos = new_pos
            for exit in exits:
                exit.color = self.intermediate_color
            # The final set of exits is also the goal, so leave that there.
            return exits
        
        # Place the objects for both games
        place_obj(self.grid, origin_cell, self.games[0])
        exits = place_obj(self.grid, exit_cell, self.games[1])

        # Change the way that agents spawn here
        # and change the spawn delay so that they
        # cannot be overwritten in the base environment
        # Modify the spawn kwargs to work
        agent_spawn_kwargs = self.agent_spawn_kwargs
        top = getattr(agent_spawn_kwargs, 'top', (0,0))
        size = getattr(agent_spawn_kwargs, 'size', (self.size, self.size))
        agent_spawn_kwargs['top'] = (top[0] + origin_cell[0], top[1] + origin_cell[1])
        agent_spawn_kwargs['size'] = (min(self.size, size[0]), min(self.size, size[1]))
        assert(top[0] < self.size and top[1] < self.size)
        for agent in self.agents:
            self.place_obj(agent, **agent_spawn_kwargs)
            agent.activate()
            agent.spawn_delay = -1

        # Set the exit color
        exits[0].color = self.exit_color
        return exits[0].pos
    
    def _get_reward(self, rwd, agent_no):
        step_rewards = np.zeros((len(self.agents, )), dtype=float)
        env_rewards = np.zeros((len(self.agents, )), dtype=float)
        env_rewards[agent_no] += rwd
        step_rewards[agent_no] += rwd

        # this is for some prestige system? not sure.
        # self.agents[agent_no].reward(rwd)

        return env_rewards, step_rewards
    # def _step_reward(self):
    #     return 1 - 0.9 * (self.step_count / self.max_steps)

    def gen_global_obs(self):
        obs = {
            'comm_act': np.stack([a.comm for a in self.agents],
                                 axis=0),  # (N, comm_len)
            'env_act': np.stack([a.env_act for a in self.agents],
                                axis=0),  # (N, 1)
        }
        # Get generator obs
        room1_obs = self.games[0].gen_room_obs()
        room2_obs = self.games[1].gen_room_obs()
        # merge the two (assume this is okay for now)
        return {**obs, **room1_obs, **room2_obs}

    def reset(self):
        # reset all agent hides
        for agent in self.agents:
            agent.hide_item_types = []
        obs_dict = MultiGridEnv.reset(self)
        obs_dict['global'] = self.gen_global_obs()
        return obs_dict

    # Modify this so that we have the agent continue to communicate useful info even when not active.
    def gen_agent_obs(self, agent, image_only=False):
        active = agent.active
        agent.active = True
        res = MultiGridEnv.gen_agent_obs(self, agent, image_only=image_only)
        agent.active = active
        return res

    def step(self, action_dict):
        obs_dict, rew_dict, _, info_dict = MultiGridEnv.step(self, action_dict)

        # Assume that the update call needs to be made
        room_rew = np.zeros((len(self.agents, )), dtype=float)
        room_info = {}
        for game in self.games:
            rew, info = game.update()
            room_rew += rew
            room_info = {**room_info, **info}

        # See if all agents made it to the goal or if we have timeout
        done = [self.agents[i].at_pos(self.goal_pos) for i in range(self.num_agents)]
        for i,d in enumerate(done):
            if d is False: continue
            # Give reward and deactivate the agent if done.
            if self.agents[i].done is False:
                self.agents[i].done = True
                rew_dict['env_rewards'][i] += 1
            self.agents[i].deactivate()
        
        success = all(done)
        timeout = (self.step_count >= self.max_steps)

        # construct return dicts
        obs_dict['global'] = self.gen_global_obs()
        
        rew_dict['env_rewards'] += room_rew
        agent_rew = {f'agent_{i}': rew_dict['env_rewards'][i] for i in range(
            len(rew_dict['env_rewards']))}
        # rew_dict = {**rew_dict, **agent_rew}
        rew_dict = agent_rew
        # if success:
        #     print(rew_dict)

        done_dict = {f'agent_{i}': done[i] or timeout for i in range(len(done))}
        done_dict['__all__'] = success or timeout
        
        env_info = {
            'done': success,
            'timeout': timeout,
            'success': success,
            'comm': obs_dict['global']['comm_act'].tolist(),
            'env_act': obs_dict['global']['env_act'].tolist(),
            't': self.step_count
        }
        # Assume the room_info and env_info are overwritable for now.
        info_dict = {**env_info, **room_info}
        return obs_dict, rew_dict, done_dict, info_dict
