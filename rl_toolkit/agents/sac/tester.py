import numpy as np
import tensorflow as tf
from dm_control import viewer

import wandb
from rl_toolkit.networks.models import Actor

from ...core.process import Process
from ...core.wrappers import dmControlGymWrapper


class Tester(Process):
    """
    Tester
    =================

    Attributes:
        env_name (str): the name of environment
        render (bool): enable the rendering
        max_steps (int): maximum number of interactions do in environment
        actor_units (list): list of the numbers of units in each Actor's layer
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        init_noise (float): initialization of the Actor's noise
        model_path (str): path to the model
        enable_wandb (bool): enable Weights & Biases logging module
    """

    def __init__(
        self,
        # ---
        env_name: str,
        render: bool,
        max_steps: int,
        frame_stack: int,
        # ---
        actor_units: list,
        clip_mean_min: float,
        clip_mean_max: float,
        init_noise: float,
        # ---
        model_path: str,
        enable_wandb: bool,
    ):
        super(Tester, self).__init__(env_name, render, frame_stack)

        self._max_steps = max_steps
        self._render = render
        self._enable_wandb = enable_wandb

        # Init actor's network
        self.model = Actor(
            units=actor_units,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            init_noise=init_noise,
        )
        self.model.build((None,) + self._env.observation_space.shape)

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.summary()

        # Init Weights & Biases
        if not self._render and self._enable_wandb:
            wandb.init(
                project="rl-toolkit",
                group=f"{env_name}",
            )
            wandb.config.max_steps = max_steps

    @tf.function(jit_compile=True)
    def policy(self, inputs):
        action = self.model(
            tf.expand_dims(inputs, axis=0),
            with_log_prob=False,
            deterministic=True,
            training=False,
        )
        return tf.squeeze(action, axis=0)

    def dm_policy(self, timestep):
        state = self._env.flatten_observation(timestep.observation)
        return self._env.scale_action(self.policy(state))

    def run(self):
        if isinstance(self._env, dmControlGymWrapper) and self._render:
            # Launch the viewer application.
            viewer.launch(self._env.env, policy=self.dm_policy)
            return

        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # Init environment
        self._last_obs, _ = self._env.reset()

        # Main loop
        while self._total_steps < self._max_steps:
            # Get the action
            action = self.policy(self._last_obs)
            action = np.array(action, copy=False, dtype=self._env.action_space.dtype)

            # Perform action
            new_obs, reward, terminated, truncated, _ = self._env.step(action)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Check the end of episode
            if terminated or truncated:
                # Logging
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print("=============================================")
                print(
                    f"Testing ... {(self._total_steps * 100) / self._max_steps} %"  # noqa
                )
                if not self._render and self._enable_wandb:
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
                self._last_obs, _ = self._env.reset()
            else:
                # Super critical !!!
                self._last_obs = new_obs
