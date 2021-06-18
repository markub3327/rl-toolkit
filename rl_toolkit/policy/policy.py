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

    def run(self):
        pass
