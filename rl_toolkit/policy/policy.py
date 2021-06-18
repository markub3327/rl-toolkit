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

    def normalise_obs(self, obs):
        # Normalize observation values to [-1, 1] range
        return obs / self._env.observation_space.high

    def run(self):
        pass
