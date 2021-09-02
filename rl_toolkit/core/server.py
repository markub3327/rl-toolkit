import numpy as np
import reverb
import tensorflow as tf

from rl_toolkit.networks.models import Actor
from rl_toolkit.utils import VariableContainer

from .process import Process


class Server(Process):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        min_replay_size (int): minimum number of samples in memory before learning starts
        max_replay_size (int): the capacity of experiences replay buffer
        samples_per_insert (float): samples per insert ratio (SPI) `= num_sampled_items / num_inserted_items`
        db_path (str): path to the database checkpoint
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        min_replay_size: int,
        max_replay_size: int,
        samples_per_insert: float,
        # ---
        db_path: str,
    ):
        super(Server, self).__init__(env_name)

        # Init actor's network
        self.actor = Actor(n_outputs=np.prod(self._env.action_space.shape))
        self.actor.build((None,) + self._env.observation_space.shape)

        # Show models details
        self.actor.summary()

        # Variables
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.uint64,
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
            db_server="localhost:8000",
            table="variables",
            variables={
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.actor.variables,
            },
        )

        # Load DB from checkpoint or make a new one
        if db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=db_path)

        if samples_per_insert is None or samples_per_insert == 0.0:
            limiter = reverb.rate_limiters.MinSize(min_replay_size)
        else:
            # 10% tolerance in rate
            samples_per_insert_tolerance = 0.1 * samples_per_insert
            error_buffer = min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=min_replay_size,
                samples_per_insert=samples_per_insert,
                error_buffer=error_buffer,
            )

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experiences",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=limiter,
                    max_size=max_replay_size,
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
                        "terminal": tf.TensorSpec([1], tf.bool),
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

        # Init variable container in DB
        self._variable_container.push_variables()

    def run(self):
        self.server.wait()

    def close(self):
        super(Server, self).close()

        # create the checkpoint of DB
        client = reverb.Client("localhost:8000")
        client.checkpoint()