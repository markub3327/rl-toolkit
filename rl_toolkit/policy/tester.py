import math

import cv2
import tensorflow as tf
import wandb

from rl_toolkit.networks.models import Actor

from .policy import Policy


class Tester(Policy):
    """
    Tester
    =================

    Attributes:
        env_name (str): the name of environment
        max_steps (int): maximum number of interactions do in environment
        render (bool): enable the rendering into the video file
        model_path (str): path to the model
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        max_steps: int,
        # ---
        render: bool = False,
        # ---
        model_path: str = None,
        log_wandb: bool = False,
    ):
        super(Tester, self).__init__(env_name)

        self._max_steps = max_steps
        self._render = render
        self._log_wandb = log_wandb

        self.actor = Actor(
            n_outputs=tf.reduce_prod(self._env.action_space.shape).numpy()
        )
        self.actor.build((None,) + self._env.observation_space.shape)

        if model_path is not None:
            self.actor.load_weights(model_path)

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
                "game.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480)
            )

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # write to stream
            if self._render:
                img_array = self._env.render(mode="rgb_array")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_array = cv2.resize(img_array, (640, 480))
                video_stream.write(img_array)

            # Get the action
            action, _ = self.actor(
                tf.expand_dims(self._last_obs, axis=0),
                with_log_prob=False,
                deterministic=True,
            )
            action = tf.squeeze(action, axis=0).numpy()

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

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
            else:
                # super critical !!!
                self._last_obs = new_obs

        # Release video file stream
        if self._render:
            video_stream.release()
