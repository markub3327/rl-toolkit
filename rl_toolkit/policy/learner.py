import os

import reverb
import tensorflow as tf
import wandb
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from rl_toolkit.networks import ActorCritic
from rl_toolkit.networks.layers import MultivariateGaussianNoise
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
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): the learning rate for Actor's optimizer
        critic_learning_rate (float): the learning rate for Critic's optimizer
        alpha_learning_rate (float): the learning rate for Alpha's optimizer
        gamma (float): the discount factor
        model_path (str): path to the model
        db_path (str): path to the database checkpoint
        save_path (str): path to the models for saving
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        max_steps: int,
        # ---
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        # ---
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # ---
        gamma: float = 0.99,
        init_alpha: float = 1.0,
        # ---
        model_path: str = None,
        db_path: str = None,
        save_path: str = None,
        # ---
        log_wandb: bool = False,
        log_interval: int = 64,
    ):
        super(Learner, self).__init__(env_name, log_wandb)

        self._max_steps = max_steps
        self._db_path = db_path
        self._save_path = save_path
        self._log_interval = log_interval

        if model_path is None:
            self.model = ActorCritic(
                num_of_outputs=tf.reduce_prod(self._env.action_space.shape).numpy(),
                gamma=gamma,
                init_alpha=init_alpha,
            )
            self.model.build((None,) + self._env.observation_space.shape)
            self.model.compile(
                actor_optimizer=Adam(learning_rate=actor_learning_rate),
                critic_optimizer=Adam(learning_rate=critic_learning_rate),
                alpha_optimizer=Adam(learning_rate=alpha_learning_rate),
            )
            print("Model created succesful ...")
        else:
            self.model = load_model(
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
                    rate_limiter=reverb.rate_limiters.MinSize(batch_size),
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

        # Initializes the reverb client and tf.dataset
        self.client = reverb.Client("localhost:8000")
        self.dataset_iterator = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address="localhost:8000",
                table="experience",
                max_in_flight_samples_per_worker=10,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.buffer_capacity = buffer_capacity
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.alpha_learning_rate = alpha_learning_rate
            wandb.config.gamma = gamma
            wandb.config.init_alpha = init_alpha

    @tf.function
    def _train(self):
        # Get data from replay buffer
        sample = self.dataset_iterator.get_next()

        # Train the Actor-Critic model
        losses = self.model.train_step(sample.data)

        # Store new actor's params
        self._variable_container.push_variables()

        return losses

    def run(self):
        while self._train_step.numpy() < self._max_steps:
            # update models
            losses = self._train()

            # log metrics
            if (self._train_step.numpy() % self._log_interval) == 0:
                print("=============================================")
                print(f"Step: {self._train_step.numpy()}")
                print(f"Alpha loss: {losses['alpha_loss']}")
                print(f"Critic loss: {losses['critic_loss']}")
                print(f"Actor loss: {losses['actor_loss']}")
                print("=============================================")
                print(
                    f"Training ... {tf.floor(self._train_step * 100 / self._max_steps)} %"  # noqa
                )

            if self._log_wandb:
                # log of epoch's mean loss
                wandb.log(
                    {
                        "Alpha": self.model.alpha,
                        "Alpha loss": losses["alpha_loss"],
                        "Critic loss": losses["critic_loss"],
                        "Actor loss": losses["actor_loss"],
                    },
                    step=self._train_step.numpy(),
                )

            # increase the training step
            self._train_step.assign_add(1)

        # Stop the agents
        self._stop_agents.assign(True)
        self._variable_container.push_variables()

    def save(self):
        if self._db_path:
            # Save database
            self.client.checkpoint()

        if self._save_path:
            # Save model
            self.model.save(os.path.join(self._save_path, "actor_critic"))

            # Save model to png
            plot_model(
                self.model,
                to_file=os.path.join(self._save_path, "actor_critic.png"),
                show_shapes=True,
                rankdir="LR",
                expand_nested=True,
            )
