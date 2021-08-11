import os

import gym
import numpy as np
import pybullet_envs  # noqa
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import wandb
from rl_toolkit.networks import ActorCritic
from rl_toolkit.utils import ReplayBuffer


class Agent:
    """
    Agent
    =================
    Attributes:
        env_name (str): the name of environment
        max_steps (int): maximum number of interactions do in environment
        env_steps (int): number of steps per rollout
        warmup_steps (int): number of interactions before using policy network
        buffer_capacity (int): the capacity of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): the learning rate for Actor's optimizer
        critic_learning_rate (float): the learning rate for Critic's optimizer
        alpha_learning_rate (float): the learning rate for Alpha's optimizer
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        save_path (str): path to the models for saving
        model_path (str): path to the model
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        max_steps: int,
        env_steps: int,
        warmup_steps: int,
        # ---
        buffer_capacity: int,
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
        save_path: str,
        model_path: str,
        # ---
        log_wandb: bool,
    ):
        # Herne prostredie
        self._env = gym.make(env_name)

        self._max_steps = max_steps
        self._env_steps = env_steps
        self._warmup_steps = warmup_steps
        self._batch_size = batch_size
        self._save_path = save_path
        self._log_wandb = log_wandb

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
            actor_optimizer=Adam(learning_rate=actor_learning_rate),
            critic_optimizer=Adam(learning_rate=critic_learning_rate),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate),
        )

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.actor.summary()
        self.model.critic.summary()

        # Init replay buffer
        self._memory = ReplayBuffer(
            obs_dim=self._env.observation_space.shape,
            act_dim=self._env.action_space.shape,
            max_size=buffer_capacity,
        )

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit", group=f"{env_name}")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.env_steps = env_steps
            wandb.config.warmup_steps = warmup_steps
            wandb.config.buffer_capacity = buffer_capacity
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.alpha_learning_rate = alpha_learning_rate
            wandb.config.gamma = gamma
            wandb.config.tau = tau
            wandb.config.init_alpha = init_alpha

    @tf.function
    def train(self, sample):
        # Train the Actor-Critic model
        losses = self.model.train_step(sample)

        return losses

    def random_policy(self, input):
        action = self._env.action_space.sample()
        return action

    @tf.function
    def collect_policy(self, input):
        action, _ = self.model.actor(
            tf.expand_dims(input, axis=0),
            with_log_prob=False,
            deterministic=False,
        )
        return tf.squeeze(action, axis=0)

    def collect(self, max_steps, policy):
        # collect the rollout
        for _ in range(max_steps):
            # Get the action
            action = policy(self._last_obs)
            action = np.array(action, dtype="float32")

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # store the interaction
            # https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py
            self._memory.store(self._last_obs, action, reward, new_obs, terminal)

            # Check the end of episode
            if terminal:
                # logovanie
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print(f"Train step: {self._train_step}")
                print("=============================================")
                if self._log_wandb:
                    wandb.log(
                        {
                            "Epoch": self._total_episodes,
                            "Score": self._episode_reward,
                            "Steps": self._episode_steps,
                        },
                        step=self._train_step,
                    )

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs = self._env.reset()
            else:
                # Super critical !!!
                self._last_obs = new_obs

    def run(self):
        # init environment
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._train_step = 0
        self._last_obs = self._env.reset()

        # zahrievacie kola
        self.collect(self._warmup_steps, self.random_policy)

        # hlavny cyklus hry
        while self._train_step < self._max_steps:
            self.collect(self._env_steps, self.collect_policy)

            # Get data from replay buffer
            data = self._memory.sample(self._batch_size)

            # update models
            losses = self.train(data)

            # log metrics
            if self._log_wandb:
                wandb.log(
                    {
                        "Log alpha": self.model.log_alpha,
                        "Alpha loss": losses["alpha_loss"],
                        "Critic loss": losses["critic_loss"],
                        "Actor loss": losses["actor_loss"],
                    },
                    step=self._train_step,
                )

            # increase the training step
            self._train_step += 1

    def save(self):
        if self._save_path:
            # Save model
            self.model.save_weights(os.path.join(self._save_path, "actor_critic.h5"))
            self.model.actor.save_weights(os.path.join(self._save_path, "actor.h5"))

    def close(self):
        # zatvor herne prostredie
        self._env.close()
