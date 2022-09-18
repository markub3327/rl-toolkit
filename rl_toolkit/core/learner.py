import os

import numpy as np
import tensorflow as tf
import reverb
import wandb
import threading
from tensorflow.keras.optimizers import Adam

from rl_toolkit.networks.models import ActorCritic, Counter
from rl_toolkit.utils import VariableContainer

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
        actor_global_clipnorm: float,
        critic_global_clipnorm: float,
        alpha_global_clipnorm: float,
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
        self._train_step = 0
        self._save_path = save_path
        self._log_interval = log_interval
        self._db_server = db_server

        # Counter
        self.target_counter = Counter(
            critic_units, gamma=gamma, target_model=None, tau=tau, beta=0.5
        )
        self.target_counter.build(
            [
                (None,) + self._env.observation_space.shape,
                (None,) + self._env.action_space.shape,
            ]
        )
        self.counter = Counter(
            critic_units,
            gamma=gamma,
            target_model=self.target_counter,
            tau=tau,
            beta=0.5,
        )
        self.counter.compile(
            optimizer=Adam(
                learning_rate=critic_learning_rate,
                global_clipnorm=critic_global_clipnorm,
            ),
        )
        self.counter.build(
            [
                (None,) + self._env.observation_space.shape,
                (None,) + self._env.action_space.shape,
            ]
        )

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
            counter=self.counter,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(
                learning_rate=actor_learning_rate, global_clipnorm=actor_global_clipnorm
            ),
            critic_optimizer=Adam(
                learning_rate=critic_learning_rate,
                global_clipnorm=critic_global_clipnorm,
            ),
            alpha_optimizer=Adam(
                learning_rate=alpha_learning_rate, global_clipnorm=alpha_global_clipnorm
            ),
        )

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.summary()
        self.counter.summary()

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

        # Initializes the reverb's dataset
        # self.dataset_iterator1 = iter(
        #     reverb.TrajectoryDataset.from_table_signature(
        #         server_address=self._db_server,
        #         table="experience_on_policy",
        #         max_in_flight_samples_per_worker=(2 * batch_size),
        #     )
        #     .batch(batch_size, drop_remainder=True)
        #     .prefetch(tf.data.AUTOTUNE)
        # )
        self.dataset_iterator1 = self.dataset_iterator2 = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address=self._db_server,
                table="experience_off_policy",
                max_in_flight_samples_per_worker=(2 * batch_size),
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
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

    # @tf.function
    # def _train_counter(self):
    #     # Get data from replay buffer
    #     sample_on_policy = self.dataset_iterator1.get_next()

    #     # Train the Counter model
    #     history = self.counter.train_step(sample_on_policy)

    #     return history

    @tf.function
    def _train_agent(self):
        # Get data from replay buffer
        sample_off_policy = self.dataset_iterator2.get_next()

        # Train the Counter model
        history2 = self.counter.train_step(sample_off_policy)

        # Train the Actor-Critic model
        history1 = self.model.train_step(sample_off_policy)

        # Store new actor's params
        self._variable_container.push_variables()

        return history1, history2

    # def train_counter(self):
    #     while self._train_step < self._train_steps:
    #         # update models
    #         history = self._train_counter()

    #         # log of epoch's mean loss
    #         wandb.log(
    #             {
    #                 "counter_loss": history["counter_loss"],
    #                 "e_value": history["e_value"],
    #             },
    #             step=self._train_step.numpy(),
    #         )

    def train_agent(self):
        while self._train_step < self._train_steps:
            # update models
            history1, history2 = self._train_agent()

            # log of epoch's mean loss
            wandb.log(
                {
                    "log_alpha": history1["log_alpha"],
                    "intrinsic_reward": history1["counter"],
                    "quantiles": history1["quantiles"],
                    "alpha_loss": history1["alpha_loss"],
                    "critic_loss": history1["critic_loss"],
                    "actor_loss": history1["actor_loss"],
                    "counter_loss": history2["counter_loss"],
                    "e_value": history2["e_value"],
                },
                step=self._train_step.numpy(),
            )

            # increase the training step
            self._train_step.assign_add(1)

    def run(self):
        self.train_agent()

        #self.t_counter = threading.Thread(name="counter", target=self.train_counter)
        #self.t_agent = threading.Thread(name="agent", target=self.train_agent)
        #self.t_counter.start()
        #self.t_agent.start()

        # Wait until training is done ...
        #self.t_agent.join()
        #self.t_counter.join()

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
