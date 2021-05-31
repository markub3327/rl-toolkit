from abc import ABC, abstractmethod

import cv2
import math
import wandb
import tensorflow as tf
import numpy as np


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
        buffer_capacity (int): the maximum size of experiences replay buffer
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
        buffer_capacity: int,
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
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._logging_wandb = logging_wandb

        # check obseration's ranges
        if np.all(np.isfinite(self._env.observation_space.low)) and np.all(
            np.isfinite(self._env.observation_space.high)
        ):
            self._normalize = self._normalize_fn

            print("Observation will be normalized !\n")
        else:
            self._normalize = lambda a: a

            print("Observation cannot be normalized !\n")

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

    def _normalize_fn(self, obs):
        # print(self._env.observation_space.low)
        # print(self._env.observation_space.high)
        # print(obs)

        # Min-max method
        obs = (obs - self._env.observation_space.low) / (
            self._env.observation_space.high - self._env.observation_space.low
        )

        # print(obs)

        return obs

    def _logging_train(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print("=============================================")
        print(
            f"Training ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
        )
        if self._logging_wandb:
            wandb.log(
                {
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
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
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
                },
                step=self._total_steps,
            )

    def train(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()
        self._last_obs = self._normalize(self._last_obs)

        # hlavny cyklus hry
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            while self._total_steps < self._max_steps:
                # Refresh actor's params
                self._update_variables()

                # re-new noise matrix before every rollouts
                self._actor_agent.reset_noise()

                # collect rollouts
                for _ in range(self._env_steps):
                    # select action randomly or using policy network
                    if self._total_steps < self._learning_starts:
                        # warmup
                        action = self._env.action_space.sample()
                    else:
                        # Get the noisy action
                        action = self._get_action(
                            self._last_obs, deterministic=False
                        ).numpy()

                    # Step in the environment
                    new_obs, reward, done, _ = self._env.step(action)
                    new_obs = self._normalize(new_obs)

                    # update variables
                    self._episode_reward += reward
                    self._episode_steps += 1
                    self._total_steps += 1

                    # Update the replay buffer
                    writer.append(
                        {
                            "obs": self._last_obs,
                            "act": action,
                            "rew": np.array([reward], dtype=np.float32),
                            "done": np.array([done], dtype=np.float32),
                        }
                    )

                    if self._episode_steps > 1:
                        writer.create_item(
                            table="experience",
                            priority=1.0,
                            trajectory={
                                "obs": writer.history["obs"][-2],
                                "act": writer.history["act"][-2],
                                "rew": writer.history["rew"][-2],
                                "obs2": writer.history["obs"][-1],
                                "done": writer.history["done"][-2],
                            },
                        )

                    # check the end of episode
                    if done:
                        self._logging_train()

                        # write the final state !!!
                        writer.append({"obs": new_obs})
                        writer.create_item(
                            table="experience",
                            priority=1.0,
                            trajectory={
                                "obs": writer.history["obs"][-2],
                                "act": writer.history["act"][-2],
                                "rew": writer.history["rew"][-2],
                                "obs2": writer.history["obs"][-1],
                                "done": writer.history["done"][-2],
                            },
                        )

                        # blocks until all the items have been sent to the server
                        writer.end_episode(timeout_ms=1000)

                        # init variables
                        self._episode_reward = 0.0
                        self._episode_steps = 0
                        self._total_episodes += 1

                        # init environment
                        self._last_obs = self._env.reset()
                        self._last_obs = self._normalize(self._last_obs)

                        # interrupt the rollout
                        break

                    # super critical !!!
                    self._last_obs = new_obs

                # update models
                if self._total_steps >= self._learning_starts:
                    self._update()
                    self._logging_models()

    def test(self, render):
        self._total_steps = 0
        self._total_episodes = 0

        # init video file
        if render:
            video_stream = cv2.VideoWriter(
                "video/game.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480)
            )

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            self._episode_reward = 0.0
            self._episode_steps = 0
            done = False

            self._last_obs = self._env.reset()
            self._last_obs = self._normalize(self._last_obs)

            # collect rollout
            while not done:
                # write to stream
                if render:
                    img_array = self._env.render(mode="rgb_array")
                    img_array = cv2.resize(img_array, (640, 480))
                    video_stream.write(img_array)

                # Get the action
                action = self._get_action(self._last_obs, deterministic=True).numpy()

                # perform action
                new_obs, reward, done, _ = self._env.step(action)
                new_obs = self._normalize(new_obs)

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

        # Release video file stream
        if render:
            video_stream.release()
