import os

import numpy as np
import reverb
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam

from rl_toolkit.networks.models import ActorCritic
from rl_toolkit.utils import VariableContainer, make_reverb_dataset

from .process import Process


class Learner(Process):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
        train_steps (int): number of training steps
        batch_size (int): size of mini-batch used for training
        actor_units (list): list of the numbers of units in each Actor's layer
        critic_units (list): list of the numbers of units in each Critic's layer
        actor_learning_rate (float): the learning rate for the Actor's optimizer
        critic_learning_rate (float): the learning rate for the Critic's optimizer
        alpha_learning_rate (float): the learning rate for the Alpha's optimizer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        init_noise (float): initialization of the Actor's noise
        model_path (str): path to the model
        save_path (str): path to the models for saving
        log_interval (int): the logging interval to the console
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        train_steps: int,
        batch_size: int,
        # ---
        actor_units: list,
        critic_units: list,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        # ---
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        # ---
        clip_mean_min: float,
        clip_mean_max: float,
        # ---
        gamma: float,
        tau: float,
        init_alpha: float,
        init_noise: float,
        # ---
        model_path: str,
        save_path: str,
        # ---
        log_interval: int,
    ):
        super(Learner, self).__init__(env_name, False)

        self._train_steps = train_steps
        self._save_path = save_path
        self._log_interval = log_interval
        self._db_server = db_server

        # Init actor-critic's network
        self.model = ActorCritic(
            actor_units=actor_units,
            critic_units=critic_units,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop=top_quantiles_to_drop,
            n_critics=n_critics,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            gamma=gamma,
            tau=tau,
            init_alpha=init_alpha,
            init_noise=init_noise,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(
                learning_rate=actor_learning_rate, global_clipnorm=40.0
            ),
            critic_optimizer=Adam(
                learning_rate=critic_learning_rate, global_clipnorm=40.0
            ),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate),
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
            db_server=self._db_server,
            table="variable",
            variables={
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.model.actor.variables,
            },
        )

        # Init variable container from DB server
        self._variable_container.update_variables()

        # Initializes the reverb's dataset
        self.dataset_iterator = iter(
            make_reverb_dataset(
                server_address=self._db_server,
                table="experience",
                batch_size=batch_size,
            )
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.train_steps = train_steps
        wandb.config.batch_size = batch_size
        wandb.config.actor_units = actor_units
        wandb.config.critic_units = critic_units
        wandb.config.actor_learning_rate = actor_learning_rate
        wandb.config.critic_learning_rate = critic_learning_rate
        wandb.config.alpha_learning_rate = alpha_learning_rate
        wandb.config.n_quantiles = n_quantiles
        wandb.config.top_quantiles_to_drop = top_quantiles_to_drop
        wandb.config.n_critics = n_critics
        wandb.config.clip_mean_min = clip_mean_min
        wandb.config.clip_mean_max = clip_mean_max
        wandb.config.gamma = gamma
        wandb.config.tau = tau
        wandb.config.init_alpha = init_alpha
        wandb.config.init_noise = init_noise

    @tf.function(jit_compile=True)
    def _step(self, data):
        # Train the Actor-Critic model
        return self.model.train_step(data)

    def run(self):
        while self._train_step < self._train_steps:
            # Get data from replay buffer
            sample = self.dataset_iterator.get_next()

            # update models
            losses = self._step(sample.data)

            # log metrics
            if (self._train_step % self._log_interval) == 0:
                print("=============================================")
                print(f"Train step: {self._train_step.numpy()}")
                print(f"Alpha loss: {losses['alpha_loss']}")
                print(f"Critic loss: {losses['critic_loss']}")
                print(f"Actor loss: {losses['actor_loss']}")
                print(f"Q value: {tf.reduce_mean(losses['quantiles'])}")
                print(f"log Alpha: {self.model.log_alpha.numpy()}")
                print("=============================================")
                print(
                    f"Training ... {(self._train_step.numpy() * 100) / self._train_steps} %"  # noqa
                )
            wandb.log(
                {
                    "Alpha loss": losses["alpha_loss"],
                    "Critic loss": losses["critic_loss"],
                    "Actor loss": losses["actor_loss"],
                    "Quantiles": wandb.Histogram(losses["quantiles"]),
                    "log Alpha": self.model.log_alpha,
                },
                step=self._train_step.numpy(),
            )

            # increase the training step
            self._train_step.assign_add(1)

            # Store new actor's params
            self._variable_container.push_variables()

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

        # create the checkpoint of the database
        client = reverb.Client(self._db_server)
        client.checkpoint()
