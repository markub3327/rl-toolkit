from rl_toolkit.networks import Actor

import cv2
import math
import wandb

import numpy as np
import tensorflow as tf


class Tester:
    """
    Soft Actor-Critic
    =================

    Paper: https://arxiv.org/pdf/1812.05905.pdf

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        model_a_path (str): path to the actor's model
        logging_wandb (bool): logging by WanDB
    """

    def __init__(
        self,
        # ---
        env,
        # ---
        max_steps: int,
        # ---
        model_a_path: str = None,
        logging_wandb: bool = False,
    ):
        self._env = env
        self._max_steps = max_steps
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

        # Actor network (for agent)
        self._actor = Actor(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            model_path=model_a_path,
        )

        # init Weights & Biases
        if self._logging_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps

    @tf.function
    def _get_action(self, state, deterministic):
        a, _ = self._actor.predict(
            tf.expand_dims(state, axis=0),
            with_logprob=False,
            deterministic=deterministic,
        )
        return tf.squeeze(a, axis=0)  # remove batch_size dim

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

    def run(self, render):
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
