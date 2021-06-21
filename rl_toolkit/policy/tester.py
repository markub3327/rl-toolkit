import math

import cv2
import tensorflow as tf
import wandb
from tensorflow.keras.models import load_model

from rl_toolkit.networks import ActorCritic
from rl_toolkit.networks.layers import MultivariateGaussianNoise

from .policy import Policy


class Tester(Policy):
    """
    Tester
    =================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        model_path (str): path to the model
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env,
        # ---
        max_steps: int,
        # ---
        render: bool = False,
        # ---
        model_path: str = None,
        log_wandb: bool = False,
    ):
        super(Tester, self).__init__(env, log_wandb)

        self._max_steps = max_steps
        self._render = render

        if model_path is None:
            self.model = ActorCritic(
                num_of_outputs=tf.reduce_prod(self._env.action_space.shape).numpy(),
                gamma=0.0,
            )
            self.model.build((None,) + self._env.observation_space.shape)
            print("Model created succesful ...")
        else:
            self.model = load_model(
                model_path,
                custom_objects={"MultivariateGaussianNoise": MultivariateGaussianNoise},
            )
            print("Model loaded succesful ...")

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps

    def run(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # init video file
        if self._render:
            video_stream = cv2.VideoWriter(
                "video/game.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480)
            )

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # write to stream
            if self._render:
                img_array = self._env.render(mode="rgb_array")
                img_array = cv2.resize(img_array, (640, 480))
                video_stream.write(img_array)

            # Get the action
            action, _ = self.model.actor(tf.expand_dims(self._last_obs, axis=0))
            action = tf.squeeze(action, axis=0).numpy()

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # super critical !!!
            self._last_obs = new_obs

            # Check the end of episode
            if terminal:
                # logovanie
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print("=============================================")
                print(
                    f"Testing ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"  # noqa
                )

                if self._log_wandb:
                    wandb.log(
                        {
                            "Epoch": self._total_episodes,
                            "Score": self._episode_reward,
                            "Steps": self._episode_steps,
                        },
                        step=self._total_steps,
                    )

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs = self._env.reset()

        # Release video file stream
        if self._render:
            video_stream.release()
