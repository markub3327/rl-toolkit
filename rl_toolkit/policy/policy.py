import numpy as np


class Policy:
    """
    Base class for policies
    =================

    Attributes:
        env: the instance of environment object
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env,
        # ---
        log_wandb: bool = False,
    ):
        self._env = env
        self._log_wandb = log_wandb

        # check obseration's ranges
        if np.all(np.isfinite(self._env.observation_space.low)) and np.all(
            np.isfinite(self._env.observation_space.high)
        ):
            self._normalize = self._normalize_observation

            print("Observation will be normalized !\n")
        else:
            self._normalize = lambda a: a

            print("Observation cannot be normalized !\n")

    def _normalize_observation(self, obs):
        # Min-max method
        return obs / self._env.observation_space.high

    def run(self):
        pass
