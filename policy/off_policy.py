import math
from abc import ABC, abstractmethod

import tensorflow as tf

import wandb

# utilities
from utils.replay_buffer import ReplayBuffer


class OffPolicy(ABC):
    """
    The base for Off-Policy algorithms
    ==================================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        env_steps (int): maximum number of steps in each rollout
        gradient_steps (int): number of update steps after each rollout
        learning_starts (int): number of interactions before using policy network
        buffer_size (int): the maximum size of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        tau (float): the soft update coefficient for target networks
        gamma (float): the discount factor
        logging_wandb (bool): logging by WanDB
    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        env_steps: int,
        gradient_steps: int,
        # ---
        learning_starts: int,
        # ---
        buffer_size: int,
        batch_size: int,
        # ---
        tau: float,
        gamma: float,
        # ---
        logging_wandb: bool,
    ):
        self._env = env
        self._max_steps = max_steps
        self._env_steps = env_steps
        self._gradient_steps = gradient_steps
        self._learning_starts = learning_starts
        self._batch_size = batch_size
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._logging_wandb = logging_wandb

        # init replay buffer
        self._rpm = ReplayBuffer(
            obs_dim=self._env.observation_space.shape,
            act_dim=self._env.action_space.shape,
            size=buffer_size,
        )

    @abstractmethod
    def _get_action(self, state, deterministic):
        ...

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    @abstractmethod
    def _update(self):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def convert(self, path):
        ...

    @abstractmethod
    def _logging_models(self):
        ...

    def _logging_train(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print(f"ReplayBuffer: {len(self._rpm)}")
        print("=============================================")
        print(
            f"Training ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
        )
        if self._logging_wandb:
            wandb.log(
                {
                    "Epoch": self._total_episodes,
                    "Score": self._episode_reward,
                    "Steps": self._episode_steps,
                    "ReplayBuffer": len(self._rpm),
                },
                step=self._total_steps,
            )

    def _logging_test(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print("=============================================")
        print(
            f"Testing ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
        )
        if self._logging_wandb:
            wandb.log(
                {
                    "Epoch": self._total_episodes,
                    "Score": self._episode_reward,
                    "Steps": self._episode_steps,
                },
                step=self._total_steps,
            )

    def _collect_rollouts(self):
        for _ in range(self._env_steps):
            # select action randomly or using policy network
            if self._total_steps < self._learning_starts:
                # warmup
                action = self._env.action_space.sample()
            else:
                # Get the noisy action
                action = self._get_action(self._last_obs, deterministic=False).numpy()

            # Step in the environment
            new_obs, reward, done, _ = self._env.step(action)

            # update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Update the replay buffer
            self._rpm.store(self._last_obs, action, reward, new_obs, done)

            # check the end of episode
            if done:
                self._logging_train()

                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # init environment
                self._last_obs = self._env.reset()

                # interrupt the rollout
                break

            # super critical !!!
            self._last_obs = new_obs

    def train(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # re-new noise matrix before every rollouts
            self._actor.reset_noise()

            # collect rollouts
            self._collect_rollouts()

            # update models
            if (
                self._total_steps >= self._learning_starts
                and len(self._rpm) >= self._batch_size
            ):
                self._update()
                self._logging_models()
                # self.convert()

    def test(self, render):
        self._total_steps = 0
        self._total_episodes = 0

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            self._episode_reward = 0.0
            self._episode_steps = 0
            done = False

            self._last_obs = self._env.reset()

            # collect rollout
            while not done:
                # Get the action
                action = self._get_action(self._last_obs, deterministic=True).numpy()

                # perform action
                new_obs, reward, done, _ = self._env.step(action)

                # update variables
                self._episode_reward += reward
                self._episode_steps += 1
                self._total_steps += 1

                # super critical !!!
                self._last_obs = new_obs

            # increment episode
            self._total_episodes += 1

            # logovanie
            self._logging_test()
