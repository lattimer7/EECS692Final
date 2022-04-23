import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import Wall

class OneRoomPuzzleMultiGrid(MultiGridEnv):
    """
    Single puzzle room environment, where the puzzles are
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
        width = self.size
        height = self.size

        super(OneRoomPuzzleMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height):
        """Generate grid without agents."""

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        # Sample a random generator
        gen_id = np.random.randint(len(self.generators))

        # Get the generator objects and update mission
        # TODO: will this update the mission?
        self.current_game = self.generators[gen_id].generate(self)
        self.mission = self.current_game.mission

        # Add all objects that are not boundary walls.
        objs, exits = self.current_game.get_objs()
        for pos, obj in objs.items():
            # TODO: Will this skip all boundary wall types?
            if isinstance(obj, Wall):
                if pos[0] != 0            \
                    and pos[1] != 0       \
                    and pos[0] != width-1 \
                    and pos[1] != height-1:
                        continue
            # Otherwise place the object into the grid.
            self.grid.set(pos[0], pos[1], obj)
            obj.pos = pos

        # We set our goal to the exit
        # self.goal_pos = exits[0]
        # print(self.goal_pos)

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
        room_obs = self.current_game.gen_room_obs()
        # merge the two (assume this is okay for now)
        return {**obs, **room_obs}

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
        room_rew, room_info = self.current_game.update()

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
