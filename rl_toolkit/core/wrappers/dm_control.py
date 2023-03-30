import collections

import gymnasium
import numpy as np
from dm_control import suite
from gymnasium import spaces


def dmControlGetTasks():
    return suite.ALL_TASKS


class dmControlGymWrapper(gymnasium.Env):
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)

        # action info
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            action_spec.minimum,
            action_spec.maximum,
            dtype=action_spec.dtype,
        )

        observation = self.env._task.get_observation(self.env._physics)
        observation = self.flatten_observation(observation)
        self.observation_space = spaces.Box(
            np.full(observation.shape, -np.inf, dtype=observation.dtype),
            np.full(observation.shape, np.inf, dtype=observation.dtype),
            dtype=observation.dtype,
        )

    def reset(self):
        time_step = self.env.reset()
        obs = self.flatten_observation(time_step.observation)
        return obs, {}

    def step(self, action):
        action = self.scale_action(action)
        time_step = self.env.step(action)
        obs = self.flatten_observation(time_step.observation)
        return (
            obs,
            time_step.reward,
            False,
            time_step.last(),
            {},
        )

    def flatten_observation(self, observation):
        if isinstance(observation, collections.OrderedDict):
            keys = observation.keys()
        else:
            # Keep a consistent ordering for other mappings.
            keys = sorted(observation.keys())

        observation_arrays = [observation[key].ravel() for key in keys]
        return np.concatenate(observation_arrays)

    def scale_action(self, action):
        # map to [0, 1]
        action = 0.5 * (action + 1.0)

        # map to [min, max]
        action *= self.action_space.high - self.action_space.low
        action += self.action_space.low

        return action
