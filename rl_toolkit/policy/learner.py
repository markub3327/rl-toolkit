import os

import numpy as np
import reverb
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import wandb
from rl_toolkit.networks import ActorCritic
from rl_toolkit.utils import VariableContainer

from .policy import Policy


class Learner(Policy):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        max_steps (int): maximum number of interactions do in environment
        buffer_capacity (int): the capacity of experiences replay buffer
        min_replay_size (int): minimum number of samples in memory before learning starts
        samples_per_insert (int): samples per insert ratio (SPI) `= num_sampled_items / num_inserted_items`
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): the learning rate for Actor's optimizer
        critic_learning_rate (float): the learning rate for Critic's optimizer
        alpha_learning_rate (float): the learning rate for Alpha's optimizer
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        model_path (str): path to the model
        db_path (str): path to the database checkpoint
        save_path (str): path to the models for saving
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        max_steps: int,
        buffer_capacity: int,
        min_replay_size: int,
        samples_per_insert: int,
        batch_size: int,
        # ---
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        # ---
        gamma: float,
        tau: float,
        init_alpha: float,
        # ---
        model_path: str,
        db_path: str,
        save_path: str,
    ):
        super(Learner, self).__init__(env_name)

        self._max_steps = max_steps
        self._save_path = save_path

        # Init actor-critic's network
        self.model = ActorCritic(
            n_quantiles=35,
            top_quantiles_to_drop=3,
            n_critics=3,
            n_outputs=np.prod(self._env.action_space.shape),
            gamma=gamma,
            tau=tau,
            init_alpha=init_alpha,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(learning_rate=actor_learning_rate, clipnorm=1.0),
            critic_optimizer=Adam(learning_rate=critic_learning_rate, clipnorm=1.0),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate, clipnorm=1.0),
        )

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.summary()

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
            "localhost",
            "variables",
            {
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.model.actor.variables,
            },
        )

        # Load DB from checkpoint or make a new one
        if db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=db_path)

        if samples_per_insert is None:
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
                    name="experience",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=limiter,
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

        # Initializes the reverb's dataset
        self.client = reverb.Client("localhost:8000")
        self.dataset_iterator = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address="localhost:8000",
                table="experience",
                max_in_flight_samples_per_worker=(2 * batch_size),
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.max_steps = max_steps
        wandb.config.buffer_capacity = buffer_capacity
        wandb.config.min_replay_size = min_replay_size
        wandb.config.samples_per_insert = samples_per_insert
        wandb.config.batch_size = batch_size
        wandb.config.actor_learning_rate = actor_learning_rate
        wandb.config.critic_learning_rate = critic_learning_rate
        wandb.config.alpha_learning_rate = alpha_learning_rate
        wandb.config.gamma = gamma
        wandb.config.tau = tau
        wandb.config.init_alpha = init_alpha

    @tf.function
    def _step(self):
        # increase the training step
        self._train_step.assign_add(1)

        # Get data from replay buffer
        sample = self.dataset_iterator.get_next()

        # Train the Actor-Critic model
        losses = self.model.train_step(sample.data)

        # Store new actor's params
        self._variable_container.push_variables()

        return losses

    def run(self):
        while self._train_step < self._max_steps:
            # update models
            losses = self._step()

            # log metrics
            wandb.log(
                {
                    "Log alpha": self.model.log_alpha,
                    "Alpha loss": losses["alpha_loss"],
                    "Critic loss": losses["critic_loss"],
                    "Actor loss": losses["actor_loss"],
                },
                step=self._train_step.numpy(),
            )

        # Stop the agents
        self._stop_agents.assign(True)
        self._variable_container.push_variables()

    def save(self):
        if self._save_path:
            # create path if not exists
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)

            # Save model
            self.model.save_weights(os.path.join(self._save_path, "actor_critic.h5"))
            self.model.actor.save_weights(os.path.join(self._save_path, "actor.h5"))

    def close(self):
        super(Learner, self).close()

        # create the checkpoint of DB
        self.client.checkpoint()
