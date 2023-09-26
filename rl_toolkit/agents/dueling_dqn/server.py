import reverb
import tensorflow as tf

from rl_toolkit.networks.models import DuelingDQN
from rl_toolkit.utils import VariableContainer

from ...core.process import Process


class Server(Process):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        port (int): the port number of database server
        actor_units (list): list of the numbers of units in each Actor's layer
        critic_units (list): list of the numbers of units in each Critic's layer
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        init_noise (float): initialization of the Actor's noise
        min_replay_size (int): minimum number of samples in memory before learning starts
        max_replay_size (int): the capacity of experiences replay buffer
        samples_per_insert (int): samples per insert ratio (SPI) `= num_sampled_items / num_inserted_items`
        actor_critic_path (str): path to the Actor-Critic model
        db_path (str): path to the database checkpoint
    """

    def __init__(
        self,
        # ---
        env_name: str,
        port: int,
        # ---
        num_layers: int,
        embed_dim: int,
        ff_mult: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        gamma: float,
        tau: float,
        # ---
        min_replay_size: int,
        max_replay_size: int,
        samples_per_insert: int,
        # ---
        model_path: str,
        db_path: str,
    ):
        super(Server, self).__init__(env_name, False)

        # Init actor-critic network
        model = DuelingDQN(
            self._env.action_space.n,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ff_mult=ff_mult,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            gamma=gamma,
            tau=tau,
        )
        model.build((None,) + self._env.observation_space.shape)

        # Show models details
        model.summary()

        # load models
        if model_path is not None:
            model.load_weights(model_path)

        # Variables
        train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.uint64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )
        stop_agents = tf.Variable(
            False,
            trainable=False,
            dtype=tf.bool,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        # Table for storing variables
        variable_container = VariableContainer(
            db_server=f"localhost:{port}",
            table="variables",
            variables={
                "policy_variables": model.variables,
                "train_step": train_step,
                "stop_agents": stop_agents,
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
                reverb.Table(  # Off-policy Replay buffer
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
                        "ext_reward": tf.TensorSpec([1], tf.float64),
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
                    signature=variable_container.signature,
                ),
            ],
            port=port,
            checkpointer=checkpointer,
        )

        # Init variable container in DB
        variable_container.push_variables()

    def run(self):
        self.server.wait()

    def close(self):
        super(Server, self).close()
        print("The database server is successfully closed! 🔥🔥🔥 Bay Bay.")
