import gym
import pybullet_envs  # noqa


class Policy:
    """
    Base class for policies
    =================

    Attributes:
        env_name (str): the name of environment
    """

    def __init__(
        self,
        # ---
        env_name: str,
    ):
        # Herne prostredie
        self._env = gym.make(env_name)

    def run(self):
        pass

    def close(self):
        # zatvor herne prostredie
        self._env.close()
