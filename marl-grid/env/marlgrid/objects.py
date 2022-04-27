from faulthandler import disable
import numpy as np
from enum import IntEnum, Enum
from gym_minigrid.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

# map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'orange': np.array([255, 165, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'cyan': np.array([0, 139, 139]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'olive': np.array([128, 128, 0]),
    'grey': np.array([100, 100, 100]),
    'worst': np.array([74, 65, 42]),
    'pink': np.array([255, 0, 189]),
    'white': np.array([255, 255, 255]),
    'prestige': np.array([255, 255, 255]),
    'shadow': np.array([35, 25, 30]),  # dark purple color for invisible cells
}

# used to map colors to integers
COLOR_TO_IDX = dict({v: k for k, v in enumerate(COLORS.keys())})

OBJECT_TYPES = []


class RegisteredObjectType(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if name not in OBJECT_TYPES:
            OBJECT_TYPES.append(cls)

        def get_recursive_subclasses():
            return OBJECT_TYPES

        cls.recursive_subclasses = staticmethod(get_recursive_subclasses)
        return cls


class WorldObj(metaclass=RegisteredObjectType):
    def __init__(self, color='worst', state=0):
        self.color = color
        self.state = state
        self.contains = None

        # some objects can have agents on top (e.g. floor, open doors, etc)
        self.agents = []

        self.pos_init = None
        self.pos = None
        self.is_agent = False

    @property
    def dir(self):
        return None

    def set_position(self, pos):
        if self.pos_init is None:
            self.pos_init = pos
        self.pos = pos

    @property
    def numeric_color(self):
        return COLORS[self.color]

    @property
    def type(self):
        return self.__class__.__name__

    def can_overlap(self):
        return False

    def can_pickup(self, agent):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, agent, pos):
        return False

    def encode(self, str_class=False):
        if bool(str_class):
            enc_class = self.type
        else:
            enc_class = self.recursive_subclasses().index(self.__class__)
        if isinstance(self.color, int):
            enc_color = self.color
        else:
            enc_color = COLOR_TO_IDX[self.color]
        return enc_class, enc_color, self.state

    def describe(self):
        return f'Obj: {self.type}({self.color}, {self.state})'

    @classmethod
    def decode(cls, type, color, state):
        if isinstance(type, str):
            cls_subclasses = {c.__name__: c for c in cls.recursive_subclasses()}
            if type not in cls_subclasses:
                raise ValueError(
                    f'Not sure how to construct a {cls} of (sub)type {type}'
                )
            return cls_subclasses[type](color, state)
        elif isinstance(type, int):
            subclass = cls.recursive_subclasses()[type]
            return subclass(color, state)

    def render(self, img):
        raise NotImplementedError

    def str_render(self, dir=0):
        return '??'


class GridAgent(WorldObj):
    def __init__(self, *args, neutral_shape, can_overlap, color='red',
                 **kwargs):
        super().__init__(*args, **{'color': color, **kwargs})
        self.metadata = {
            'color': color,
        }
        self.is_agent = True
        self.comm = 0
        self.neutral_shape = neutral_shape

        self._can_overlap = can_overlap

    @property
    def dir(self):
        return self.state % 4

    @property
    def type(self):
        return 'Agent'

    @dir.setter
    def dir(self, dir):
        self.state = self.state // 4 + dir % 4

    def str_render(self, dir=0):
        return ['>>', 'VV', '<<', '^^'][(self.dir + dir) % 4]

    def can_overlap(self):
        return self._can_overlap

    def render(self, img):
        if self.neutral_shape:
            shape_fn = point_in_circle(0.5, 0.5, 0.31)
        else:
            shape_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50),
                                         (0.12, 0.81),)
            shape_fn = rotate_fn(shape_fn, cx=0.5, cy=0.5,
                                 theta=1.5 * np.pi * self.dir)
        fill_coords(img, shape_fn, COLORS[self.color])


class BulkObj(WorldObj, metaclass=RegisteredObjectType):
    def __hash__(self):
        return hash((self.__class__, self.color, self.state,
                     tuple(self.agents)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class BonusTile(WorldObj):
    def __init__(self, reward, penalty=-0.1, bonus_id=0, n_bonus=1,
                 initial_reward=True, reset_on_mistake=False, color='yellow',
                 *args, **kwargs):
        super().__init__(*args, **{'color': color, **kwargs, 'state': bonus_id})
        self.reward = reward
        self.penalty = penalty
        self.n_bonus = n_bonus
        self.bonus_id = bonus_id
        self.initial_reward = initial_reward
        self.reset_on_mistake = reset_on_mistake

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return 'BB'

    def get_reward(self, agent):
        # If the agent hasn't hit any bonus tiles, set its bonus state so that
        #  it'll get a reward from hitting this tile.
        first_bonus = False
        if agent.bonus_state is None:
            agent.bonus_state = (self.bonus_id - 1) % self.n_bonus
            first_bonus = True

        if agent.bonus_state == self.bonus_id:
            # This is the last bonus tile the agent hit
            rew = -np.abs(self.penalty)
        elif (agent.bonus_state + 1) % self.n_bonus == self.bonus_id:
            # The agent hit the previous bonus tile before this one
            agent.bonus_state = self.bonus_id
            # rew = agent.bonus_value
            rew = self.reward
        else:
            # The agent hit any other bonus tile before this one
            rew = -np.abs(self.penalty)

        if self.reset_on_mistake:
            agent.bonus_state = self.bonus_id

        if first_bonus and not bool(self.initial_reward):
            return 0
        else:
            return rew

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, reward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def can_overlap(self):
        return True

    def get_reward(self, agent):
        return self.reward

    def str_render(self, dir=0):
        return 'GG'

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Wall(BulkObj):
    def see_behind(self):
        return False

    def str_render(self, dir=0):
        return 'WW'

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class FreeDoor(WorldObj):
    class states(IntEnum):
        open = 1
        closed = 2
    
    def __init__(self, reward = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.rewarded_agents = []

    def get_reward(self, agent):
        if agent not in self.rewarded_agents:
            self.rewarded_agents.append(agent)
            return self.reward
        else:
            return 0

    def is_open(self):
        return self.state == self.states.open

    def can_overlap(self):
        # Change this so the agents can actually go through the door after open
        return self.is_open()

    def see_behind(self):
        return self.is_open()

    def toggle(self, agent, pos):
        if self.state == self.states.closed:
            self.state = self.states.open
        elif self.state == self.states.open:
            # door can only be opened once
            pass
        else:
            raise ValueError(f'?!?!?! FreeDoor in state {self.state}')
        return True

    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

# This is a special door object that is only unlocked via an environment call.
class EnvLockedDoor(FreeDoor):
    class states(IntEnum):
        open = 1
        closed = 2
        locked = 3
    
    def __init__(self, reward=0.5, require_open=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.require_open = require_open
        self.last_toggling_agent = None
        self.opening_agent = None
    
    def toggle(self, agent, pos):
        if self.state == self.states.closed:
            self.state = self.states.open
            self.opening_agent = agent
        elif self.state == self.states.open or self.state == self.states.locked:
            # door can only be opened once
            pass
        else:
            raise ValueError(f'?!?!?! EnvLockedDoor in state {self.state}')
        self.last_toggling_agent = agent
        return True

    def unlock(self):
        if self.state == self.states.locked:
            self.state = self.states.closed if self.require_open else self.states.open
        elif self.state == self.states.open or self.state == self.states.closed:
            # door can only be opened once
            pass
        else:
            raise ValueError(f'?!?!?! EnvLockedDoor in state {self.state}')
    
    def lock(self):
        if self.state == self.states.locked:
            pass
        elif self.state == self.states.open or self.state == self.states.closed:
            self.state = self.states.locked
        else:
            raise ValueError(f'?!?!?! EnvLockedDoor in state {self.state}')

    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle only if we are closed
        if self.state == self.states.closed:
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class PressurePlate(WorldObj):
    class states(IntEnum):
        novel = 1
        found = 2
        active = 3

    def __init__(self, reward=0.5, color='orange', state=states.novel, *args, **kwargs):
        super().__init__(*args, **{'color': color, 'state': state, **kwargs})
        self.reward = reward
        self.state = state

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return 'PP'

    def get_reward(self, agent):
        # If this pressure plate has already been activated, don't give a reward
        if self.state == self.states.found or self.state == self.states.active:
            return 0
        else:
            # Otherwise make it found and get a small reward.
            self.state = self.states.found
            return self.reward

    def set_active(self):
        if self.state == self.states.found:
            self.state = self.states.active

    def unset_active(self):
        if self.state == self.states.active:
            self.state = self.states.found

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


# A color cycler object
class ColorCyclerBox(WorldObj):
    colors_states = [
        'orange',
        'green',
        'cyan',
        'purple',
        'pink',
        'white',
    ]
    def __init__(self, reward=0.5, color='orange', interactable_agents = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the color state
        self.color = color
        idx = next(i for i,c in enumerate(self.colors_states) if c == color)
        self.state = idx
        if interactable_agents is not None:
            self.interactable_agents = interactable_agents
        else: 
            self.interactable_agents = []
        self.last_toggling_agent = None
        self.reward = reward
        self.disabled = False

    def disable(self):
        self.disabled = True

    def _get_next_color(self):
        max = len(self.colors_states)
        next = (self.state + 1) % max
        return next
    
    def _get_color(self):
        key = self.colors_states[self.state]
        return (key, COLORS[key])

    def toggle(self, agent, pos):
        # See if this is an agent we can actually interact with
        if not self.disabled and agent in self.interactable_agents:
            self.last_toggling_agent = agent
            self.state = self._get_next_color()
            self.color = self._get_color()[0]
            return True
        else:
            return False
        
    def can_overlap(self):
        return True
    
    def str_render(self, dir=0):
        return 'CC'
    
    def render(self, img):
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), self._get_color()[1])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.1), (0,0,0))


# A triangle key object
class StaticCodedTriangle(WorldObj):
    def __init__(self, color='orange', dir=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the color state
        self.color = color
        self.state = dir

    @property
    def dir(self):
        return self.state % 4

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return 'CT'

    def render(self, img):
        outer_fn = point_in_triangle((0.24, 0.2), (0.76, 0.50), (0.24, 0.8))
        outer_fn = rotate_fn(outer_fn, cx=0.5, cy=0.5, theta=1.5 * np.pi * self.dir)
        inner_fn = point_in_triangle((0.4135, 0.4), (0.5865, 0.50), (0.4135, 0.6))
        inner_fn = rotate_fn(inner_fn, cx=0.5, cy=0.5, theta=1.5 * np.pi * self.dir)
        fill_coords(img, outer_fn, COLORS[self.color])
        fill_coords(img, inner_fn, (0,0,0))


# A toggleable lever that can be used in place of doors
# for redbluedoors to make it chainable
class Lever(WorldObj):
    class states(IntEnum):
        on = 1
        off = 2
        disabled = 3
    
    def __init__(self, reward=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.frozen = False
        self.last_toggling_agent = None
        self.state = self.states.off
    
    def freeze(self):
        self.frozen = True

    def disable(self):
        self.state = self.states.disabled

    def enable(self):
        self.state = self.states.off
    
    def can_overlap(self):
        return False
    
    def toggle(self, agent, pos):
        if (self.state == self.states.disabled) or self.frozen:
            return False
        if self.state == self.states.off:
            self.state = self.states.on
        elif self.state == self.states.on:
            self.state = self.states.off
        else:
            raise ValueError(f'?!?!?! Lever in state {self.state}')
        self.last_toggling_agent = agent
        return True
        
    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.on:
            # Draw an I
            fill_coords(img, point_in_rect(0.44, 0.56, 0.00, 1.00), c)
        else:
            # Draw an O
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.45), c)
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.30), (0,0,0))
        if self.state == self.states.disabled:
            # Draw an X
            x_fn = point_in_rect(0.40, 0.60, 0.00, 1.00)
            fill_coords(img, rotate_fn(x_fn, cx=0.5, cy=0.5, theta=np.pi/4), COLORS['red'])
            fill_coords(img, rotate_fn(x_fn, cx=0.5, cy=0.5, theta=3*np.pi/4), COLORS['red'])


# Key object imported back from the original marlgrid
class Key(WorldObj):
    unique_id = 0
    id_wraparound = 100

    def __init__(self, interactable_agents = [], *args, **kwargs):
        self.unique_id = Key.unique_id = (Key.unique_id + 1) % Key.id_wraparound
        super().__init__(*args, **kwargs)
        if interactable_agents is not None:
            self.interactable_agents = interactable_agents
        else: 
            self.interactable_agents = []
        
    @property
    def type(self):
        return self.__class__.__name__ + str(self.unique_id)

    def can_pickup(self, agent):
        if len(self.interactable_agents) > 0:
            return agent in self.interactable_agents
        else:
            return True
        
    def can_overlap(self):
        return True

    # Clears the interactable agents so that any agent can interact
    def reveal(self):
        self.interactable_agents = []

    def str_render(self, dir=0):
        return "KK"

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


# Just locked door that can require any key object.
# This one is unlocked by the agents, then is opened.
# If required_open is true, then it will require an additional
# open step, otherwise it will just open immediately from locked
class LockedDoor(EnvLockedDoor):
    def __init__(self, key_obj=None, reward=0.5, require_open=False, *args, **kwargs):
        super().__init__(reward, require_open, *args, **kwargs)
        self.key_obj = key_obj
        self.unlocking_agent = None
        self.opening_agent = None
        self.last_toggling_agent = None
    
    def toggle(self, agent, pos):
        if self.state == self.states.locked:  # is locked
            # If the agent is carrying a key of matching color
            if (agent.carrying is not None
                    and agent.carrying == self.key_obj):
                self.state = self.states.closed if self.require_open else self.states.open
                # CONSUME ZA KEY
                agent.carrying = None
                self.unlocking_agent = agent
        elif self.state == self.states.closed:  # is unlocked but closed
            self.state = self.states.open
            self.opening_agent = agent
        elif self.state == self.states.open:  # is open
            pass
        self.last_toggling_agent = agent
        return True

    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.state == self.states.locked and self.key_obj is not None:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


# TODO: Make keyhole object
class KeyHole(WorldObj):
    unique_id = 0
    id_wraparound = 100

    class states(IntEnum):
        unlocked = 1
        locked = 2
    
    def __init__(self, key_obj=None, reward=0.5, interactable_agents = [], overlap = True, *args, **kwargs):
        self.unique_id = KeyHole.unique_id = (KeyHole.unique_id + 1) % KeyHole.id_wraparound
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.key_obj = key_obj
        self.unlocking_agent = None
        self.last_toggling_agent = None
        if interactable_agents is not None:
            self.interactable_agents = interactable_agents
        else: 
            self.interactable_agents = []
        self.overlap = overlap
        self.state = KeyHole.states.locked
    
    @property
    def type(self):
        return self.__class__.__name__ + str(self.unique_id)

    def can_overlap(self):
        return self.overlap
    
    def toggle(self, agent, pos):
        if not (agent in self.interactable_agents):
            return False
        if self.state == self.states.locked:  # is locked
            # If the agent is carrying a key of matching color
            if (agent.carrying is not None
                    and agent.carrying == self.key_obj):
                self.state = self.states.unlocked
                agent.carrying = None  # CONSUME
                self.unlocking_agent = agent
        elif self.state == self.states.unlocked:  # is open
            pass
        self.last_toggling_agent = agent
        return True

    # Draw small slot
    def render(self, img):
        c = COLORS[self.color]
        
        fill_coords(img, point_in_rect(0.20, 0.80, 0.20, 0.80), c)
        fill_coords(img, point_in_rect(0.26, 0.74, 0.26, 0.74), (0,0,0))
        
        # Draw box
        if self.state == self.states.locked and self.key_obj is not None:
            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        # Don't draw key slot if done.

