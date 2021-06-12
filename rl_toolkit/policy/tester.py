from rl_toolkit.networks.layers import Actor
from rl_toolkit.policy import Policy

import cv2
import math
import wandb

import tensorflow as tf


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

        # Actor network (for agent)
        # self._actor = Actor(
        #    state_shape=self._env.observation_space.shape,
        #    action_shape=self._env.action_space.shape,
        #    model_path=model_a_path,
        # )
        input_layer = tf.keras.layers.Input(shape=self._env.observation_space.shape)
        self.output_layer = Actor(
            num_of_outputs=tf.reduce_prod(self._env.action_space.shape)
        )(input_layer)
        self.model = tf.keras.Model(inputs=input_layer, outputs=self.output_layer)

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps

    def _log_test(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print("=============================================")
        print(
            f"Testing ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
        )

        if self._log_wandb:
            wandb.log(
                {
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
                },
                step=self._total_steps,
            )

    def run(self):
        self._total_steps = 0
        self._total_episodes = 0

        # init video file
        if self._render:
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
                if self._render:
                    img_array = self._env.render(mode="rgb_array")
                    img_array = cv2.resize(img_array, (640, 480))
                    video_stream.write(img_array)

                # Get the action
                action, _ = self.model(
                    self.model(tf.expand_dims(self._last_obs, axis=0))
                )
                action = tf.squeeze(action, axis=0).numpy()

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
            self._log_test()

        # Release video file stream
        if self._render:
            video_stream.release()
