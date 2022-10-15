import gym
from dm_control import suite


def dmControlGetTasks():
    return suite.ALL_TASKS


class dmControlGymWrapper(gym.Wrapper):
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)

    def reset(self):
        time_step = self.env.reset()
        return time_step.observation, {}

    def step(self, actions):
        time_step = self.env.step(actions)
        return time_step.observation, time_step.reward, time_step.last(), False, {}
