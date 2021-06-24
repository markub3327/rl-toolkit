import gym


class Policy:
    """
    Base class for policies
    =================

    Attributes:
        env_name (str): the name of environment
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        log_wandb: bool = False,
    ):
        self._log_wandb = log_wandb

        # Herne prostredie
        self._env = gym.make(env_name)

    def run(self):
        pass

    def close(self):
        # zatvor herne prostredie
        self._env.close()
