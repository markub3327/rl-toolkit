import tensorflow as tf

from .wrappers import FrameStack, dmControlGetTasks, dmControlGymWrapper


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
        frame_stack: int,
    ):
        # Init environment
        if any(x[0] in env_name and x[1] in env_name for x in dmControlGetTasks()):
            s = env_name.split("-")
            self._env = dmControlGymWrapper(domain_name=s[0], task_name=s[1])
        else:
            # Import third-party environments
            try:
                import flappy_bird_gymnasium  # noqa
            except ImportError as e:
                print(f"The third-party environment {e} is not available!")

            import gymnasium

            self._env = gymnasium.make(
                env_name, render_mode="human" if render else None
            )
            if frame_stack > 1:
                self._env = FrameStack(self._env, frame_stack)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices("GPU")
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def run(self):
        pass

    def save(self):
        pass

    def close(self):
        # Safely close the environment
        self._env.close()
