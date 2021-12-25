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
        port (int): the port number of database server
        actor_units (list): list of the numbers of units in each Actor's layer
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        init_noise (float): initialization of the Actor's noise
        min_replay_size (int): minimum number of samples in memory before learning starts
        max_replay_size (int): the capacity of experiences replay buffer
        samples_per_insert (int): samples per insert ratio (SPI) `= num_sampled_items / num_inserted_items`
        db_path (str): path to the database checkpoint
    """

    def __init__(
        self,
        # ---
        env_name: str,
        port: int,
        # ---
        actor_units: list,
        clip_mean_min: float,
        clip_mean_max: float,
        init_noise: float,
        # ---
        min_replay_size: int,
        max_replay_size: int,
        samples_per_insert: int,
        # ---
        db_path: str,
    ):
        super(Server, self).__init__(env_name, False)
        self._port = port

        # Init actor's network
        self.actor = Actor(
            units=actor_units,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            init_noise=init_noise,
        )
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
            db_server=f"localhost:{self._port}",
            table="variable",
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

        if samples_per_insert:
            # 10% tolerance in rate
            samples_per_insert_tolerance = 0.1 * samples_per_insert
            error_buffer = min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=min_replay_size,
                samples_per_insert=samples_per_insert,
                error_buffer=error_buffer,
            )
        else:
            limiter = reverb.rate_limiters.MinSize(min_replay_size)

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experience",
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
                    name="variable",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature=self._variable_container.signature,
                ),
            ],
            port=self._port,
            checkpointer=checkpointer,
        )

        # Init variable container in DB
        self._variable_container.push_variables()

    def run(self):
        self.server.wait()

    def close(self):
        super(Server, self).close()
        print("The database server is successfully closed! 🔥🔥🔥 Bay Bay.")
