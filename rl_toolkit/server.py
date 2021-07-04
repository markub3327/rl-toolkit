import reverb
import tensorflow as tf
from tensorflow.keras.models import load_model

from rl_toolkit.networks.layers import Actor, MultivariateGaussianNoise
from rl_toolkit.utils import VariableContainer


class Server:
    """
    Server
    =================

    Attributes:
        env_name (str): the name of environment
        buffer_capacity (int): the capacity of experiences replay buffer
        model_path (str): path to the model
        db_path (str): path to the database checkpoint
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        buffer_capacity: int = 1000000,
        # ---
        model_path: str = None,
        db_path: str = None,
    ):
        self._db_path = db_path

        if model_path is None:
            self.actor = Actor(
                num_of_outputs=tf.reduce_prod(self._env.action_space.shape).numpy()
            )
            self.actor.build((None,) + self._env.observation_space.shape)
            print("Model created succesful ...")
        else:
            self.actor = load_model(
                model_path,
                custom_objects={"MultivariateGaussianNoise": MultivariateGaussianNoise},
            )
            print("Model loaded succesful ...")

        # Show models details
        self.model.summary()

        # Variables
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )
        self._stop_agents = tf.Variable(
            False,
            trainable=False,
            dtype=tf.bool,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        # Table for storing variables
        self._variable_container = VariableContainer(
            "localhost",
            "variables",
            {
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.model.actor.variables,
            },
        )

        # load db from checkpoint or make a new one
        if self._db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=self._db_path)

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experience",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(10000),
                    max_size=buffer_capacity,
                    max_times_sampled=0,
                    signature={
                        "observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "action": tf.TensorSpec(
                            [*self._env.action_space.shape],
                            self._env.action_space.dtype,
                        ),
                        "reward": tf.TensorSpec([1], tf.float32),
                        "next_observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "terminal": tf.TensorSpec([1], tf.float32),
                    },
                ),
                reverb.Table(  # Variables container
                    name="variables",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature=self._variable_container.signature,
                ),
            ],
            port=8000,
            checkpointer=checkpointer,
        )

        # init variable container in DB
        self._variable_container.push_variables()

        self.client = reverb.Client("localhost:8000")

    def run(self):
        self.server.wait()

    def close(self):
        if self._db_path:
            # Save database
            self.client.checkpoint()
