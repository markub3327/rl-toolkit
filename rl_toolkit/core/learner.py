import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import reverb
import wandb
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
        max_steps (int): maximum number of interactions do in environment
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): the learning rate for Actor's optimizer
        critic_learning_rate (float): the learning rate for Critic's optimizer
        alpha_learning_rate (float): the learning rate for Alpha's optimizer
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
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
        max_steps: int,
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
        save_path: str,
        # ---
        log_interval: int,
    ):
        super(Learner, self).__init__(env_name)

        self._max_steps = max_steps
        self._save_path = save_path
        self._log_interval = log_interval
        self._db_server = f"{db_server}:8000"

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
            db_server=self._db_server,
            table="variables",
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
                table="experiences",
                batch_size=batch_size,
            )
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.max_steps = max_steps
        wandb.config.batch_size = batch_size
        wandb.config.actor_learning_rate = actor_learning_rate
        wandb.config.critic_learning_rate = critic_learning_rate
        wandb.config.alpha_learning_rate = alpha_learning_rate
        wandb.config.gamma = gamma
        wandb.config.tau = tau
        wandb.config.init_alpha = init_alpha

    @tf.function(jit_compile=True)
    def _step(self, data):
        # Train the Actor-Critic model
        return self.model.train_step(data)

    def run(self):
        while self._train_step < self._max_steps:
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
                print("=============================================")
                print(
                    f"Training ... {(self._train_step.numpy() * 100) / self._max_steps} %"  # noqa
                )
            wandb.log(
                {
                    "Log alpha": self.model.log_alpha,
                    "Alpha loss": losses["alpha_loss"],
                    "Critic loss": losses["critic_loss"],
                    "Actor loss": losses["actor_loss"],
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
            
            # Save model to cloud
            wandb.save(os.path.join(self._save_path, "actor_critic.h5"))
            wandb.save(os.path.join(self._save_path, "actor.h5"))
    
    def close(self):
        super(Learner, self).close()

        # create the checkpoint of DB
        client = reverb.Client(self._db_server)
        client.checkpoint()
        client.reset(table="variables")
        client.reset(table="experiences")