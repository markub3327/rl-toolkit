import os

import gym
import numpy as np
import pybullet_envs  # noqa
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam

from rl_toolkit.networks import ActorCritic
from rl_toolkit.utils.replay_buffer import ReplayBuffer


class Training:
    """
    Training
    =================

    Attributes:
        env_name (str): the name of environment
        model_path (str): path to the model
        max_steps (int): maximum number of interactions do in environment
        warmup_steps (int): number of interactions before using policy network
        env_steps (int): number of steps per rollout
        buffer_capacity (int): the capacity of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): the learning rate for Actor's optimizer
        critic_learning_rate (float): the learning rate for Critic's optimizer
        alpha_learning_rate (float): the learning rate for Alpha's optimizer
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        save_path (str): path to the models for saving
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        max_steps: int = 1000000,
        warmup_steps: int = 10000,
        env_steps: int = 64,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        # ---
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # ---
        gamma: float = 0.99,
        tau: float = 0.01,
        init_alpha: float = 1.0,
        # ---
        model_path: str = None,
        save_path: str = None,
        # ---
        log_wandb: bool = False,
    ):
        self._env = gym.make(env_name)
        self._max_steps = max_steps
        self._env_steps = env_steps
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._save_path = save_path
        self._log_wandb = log_wandb

        # Init actor-critic's network
        self.model = ActorCritic(
            n_quantiles=35,
            top_quantiles_to_drop=3,
            n_critics=3,
            n_outputs=tf.reduce_prod(self._env.action_space.shape).numpy(),
            gamma=gamma,
            tau=tau,
            init_alpha=init_alpha,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(learning_rate=actor_learning_rate, clipnorm=0.5),
            critic_optimizer=Adam(learning_rate=critic_learning_rate, clipnorm=0.5),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate, clipnorm=0.5),
        )

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.actor.summary()
        self.model.critic.summary()

        # Init replay buffer
        self._rpm = ReplayBuffer(
            state_dim=self._env.observation_space.shape,
            action_dim=self._env.action_space.shape,
            max_size=buffer_capacity,
        )

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit", group=f"{env_name}")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.warmup_steps = warmup_steps
            wandb.config.env_steps = env_steps
            wandb.config.buffer_capacity = buffer_capacity
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.alpha_learning_rate = alpha_learning_rate
            wandb.config.gamma = gamma
            wandb.config.init_alpha = init_alpha

    def random_policy(self, input):
        action = self._env.action_space.sample()
        return action

    @tf.function
    def collect_policy(self, input):
        action, _ = self.actor(
            tf.expand_dims(input, axis=0),
            with_log_prob=False,
            deterministic=False,
        )
        return tf.squeeze(action, axis=0)

    @tf.function
    def _train(self, sample):
        # Train the Actor-Critic model
        losses = self.model.train_step(sample)

        return losses

    def collect(self, max_steps, policy):
        # collect the rollout
        for _ in range(max_steps):
            # Get the action
            action = policy(self._last_obs)
            action = np.array(action, dtype="float32")

            # perform action
            next_state, reward, done, _ = self._env.step(action)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            self._rpm.store(self._state, action, reward, done)

            # Check the end of episode
            if done:
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
                        step=self._,
                    )

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs = self._env.reset()
            else:
                # Super critical !!!
                self._state = next_state

    def run(self):
        # init environment
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._state = self._env.reset()
        self._train_step = 0

        # zahrievacie kola
        self.collect(self._warmup_steps, self.random_policy)

        for train_step in range(self._max_steps):
            # re-new noise matrix before every rollouts
            self.model.actor.reset_noise()

            # collect rollouts
            self.collect(self._env_steps, self.collect_policy)

            # Get sample
            sample = self._rpm.sample(batch_size=self._batch_size)

            # update models
            losses = self._train(steps=self._env_steps, samples=sample)

            # log metrics
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

    def save(self):
        if self._save_path:
            # Save model
            self.model.save_weights(os.path.join(self._save_path, "actor_critic.h5"))
            self.model.actor.save_weights(os.path.join(self._save_path, "actor.h5"))

    def close(self):
        # zatvor herne prostredie
        self._env.close()
