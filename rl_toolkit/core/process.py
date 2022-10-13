import gym
import pybullet_envs  # noqa

import tensorflow as tf

class Process:
    """
    Base class for processes
    =================

    Attributes:
        env_name (str): the name of environment
        render (bool): enable the rendering
    """

    def __init__(
        self,
        # ---
        env_name: str,
        render: bool,
    ):
        # Herne prostredie
        self._env = gym.make(env_name, render_mode="human" if render else None)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def run(self):
        pass

    def close(self):
        # zatvor herne prostredie
        self._env.close()
