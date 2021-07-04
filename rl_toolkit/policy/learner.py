import os

import reverb
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from rl_toolkit.networks import ActorCritic
from rl_toolkit.utils import VariableContainer

from .policy import Policy


class Learner(Policy):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
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
        db_server: str,
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
        save_path: str = None,
        # ---
        log_wandb: bool = False,
        log_interval: int = 1000,
    ):
        super(Learner, self).__init__(env_name)

        self._max_steps = max_steps
        self._save_path = save_path
        self._log_interval = log_interval
        self._log_wandb = log_wandb

        self.model = ActorCritic(
            n_quantiles=35,
            top_quantiles_to_drop=3,
            n_critics=3,
            n_outputs=tf.reduce_prod(self._env.action_space.shape).numpy(),
            gamma=gamma,
            init_alpha=init_alpha,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(learning_rate=actor_learning_rate),
            critic_optimizer=Adam(learning_rate=critic_learning_rate),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate),
        )

        # Show models details
        self.model.actor.summary()
        self.model.critic.summary()

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
            db_server,
            "variables",
            {
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.model.actor.variables,
            },
        )

        # load content of variables & re-new noise matrix
        self._variable_container.update_variables()
        self.model.actor.reset_noise()

        # Initializes the reverb's dataset
        self.dataset_iterator = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address=f"{db_server}:8000",
                table="experience",
                max_in_flight_samples_per_worker=(2 * batch_size),
            )
            .batch(batch_size, drop_remainder=True)
            .cache()
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
        while self._train_step < self._max_steps:
            # update models
            losses = self._train()

            # log metrics
            if (self._train_step % self._log_interval) == 0:
                print("=============================================")
                print(f"Train step: {self._train_step.numpy()}")
                print(f"Alpha loss: {losses['alpha_loss']}")
                print(f"Critic loss: {losses['critic_loss']}")
                print(f"Actor loss: {losses['actor_loss']}")
                print("=============================================")
                print(
                    f"Training ... {(self._train_step * 100) // self._max_steps} %"  # noqa
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
        if self._save_path:
            # Save model
            self.model.save(os.path.join(self._save_path, "actor_critic"))
            self.model.actor.save(os.path.join(self._save_path, "actor"))

            # Convert the model to TF Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model.actor)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(os.path.join(self._save_path, "actor.tflite"), "wb") as f:
                f.write(tflite_model)

            # Save model to png
            plot_model(
                self.model.actor,
                to_file=os.path.join(self._save_path, "actor.png"),
                show_shapes=True,
                rankdir="LR",
                expand_nested=True,
            )
            plot_model(
                self.model.critic,
                to_file=os.path.join(self._save_path, "critic.png"),
                show_shapes=True,
                rankdir="LR",
                expand_nested=True,
            )
