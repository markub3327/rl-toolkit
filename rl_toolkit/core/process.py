import gym
import pybullet_envs  # noqa


class Process:
    """
    Base class for processes
    =================

    Attributes:
        env_name (str): the name of environment
        render (bool): enable the rendering into the video file
    """

    def __init__(
        self,
        # ---
        env_name: str,
        render: bool,
    ):
        # Herne prostredie
        self._env = gym.make(env_name)

        if render:
            self._env = gym.wrappers.RecordVideo(self._env, f"videos/{env_name}")

    def run(self):
        pass

    def close(self):
        # zatvor herne prostredie
        self._env.close()
