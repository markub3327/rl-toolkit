import os

import numpy as np
import reverb
import tensorflow as tf
import wandb

from rl_toolkit.networks.models import Actor
from rl_toolkit.utils import VariableContainer

from .process import Process


class Agent(Process):
    """
    Agent
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
        actor_units (list): list of the numbers of units in each Actor's layer
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        init_noise (float): initialization of the Actor's noise
        warmup_steps (int): number of interactions before using policy network
        env_steps (int): number of steps per rollout
        save_path (str): path to the models for saving
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        actor_units: list,
        clip_mean_min: float,
        clip_mean_max: float,
        init_noise: float,
        # ---
        warmup_steps: int,
        env_steps: int,
        # ---
        save_path: str,
    ):
        super(Agent, self).__init__(env_name, False)

        self._env_steps = env_steps
        self._warmup_steps = warmup_steps
        self._save_path = save_path

        if self._env.unwrapped.spec is not None and self._env.unwrapped.spec.id == "HumanoidRobot-v0":
            self._env.connect()

        # Init actor's network
        self.model = Actor(
            units=actor_units,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            init_noise=init_noise,
        )
        self.model.build((None,) + self._env.observation_space.shape)

        # Show models details
        self.model.summary()

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
            db_server=db_server,
            table="variables",
            variables={
                "policy_variables": self.model.variables,
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
            },
        )

        # Initializes the reverb client
        self.client = reverb.Client(db_server)

        # Init Weights & Biases
        wandb.init(
            project="rl-toolkit",
            group=f"{env_name}",
        )
        wandb.config.warmup_steps = warmup_steps
        wandb.config.env_steps = env_steps

    def random_policy(self, inputs):
        action = self._env.action_space.sample()
        return action

    @tf.function(jit_compile=True)
    def collect_policy(self, inputs):
        action = self.model(
            tf.expand_dims(inputs, axis=0),
            with_log_prob=False,
            deterministic=False,
            training=False,
        )
        return tf.squeeze(action, axis=0)

    def collect(self, writer, max_steps, policy):
        # Collect the rollout
        for _ in range(max_steps):
            # Get the action
            action = policy(self._last_obs)
            action = np.array(action, copy=False, dtype=self._env.action_space.dtype)

            # Perform action
            new_obs, ext_reward, terminated, truncated, _ = self._env.step(action)

            # Update variables
            self._episode_reward += ext_reward
            self._episode_steps += 1
            self._total_steps += 1

            # Update the replay buffer
            writer.append(
                {
                    "observation": self._last_obs,
                    "action": action,
                    "ext_reward": np.array([ext_reward], dtype=np.float64),
                    "terminal": np.array([terminated]),
                }
            )

            # Enough samples to store in the database
            if self._episode_steps > 1:
                writer.create_item(
                    table="experience",
                    priority=1.0,
                    trajectory={
                        "observation": writer.history["observation"][-2],
                        "action": writer.history["action"][-2],
                        "ext_reward": writer.history["ext_reward"][-2],
                        "next_observation": writer.history["observation"][-1],
                        "terminal": writer.history["terminal"][-2],
                    },
                )

            # Check the end of episode
            if terminated or truncated:
                # Write the final interaction !!!
                writer.append(
                    {
                        "observation": new_obs,
                    }
                )
                writer.create_item(
                    table="experience",
                    priority=1.0,
                    trajectory={
                        "observation": writer.history["observation"][-2],
                        "action": writer.history["action"][-2],
                        "ext_reward": writer.history["ext_reward"][-2],
                        "next_observation": writer.history["observation"][-1],
                        "terminal": writer.history["terminal"][-2],
                    },
                )

                # Block until all the items have been sent to the server
                writer.end_episode()

                # Logging
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print(f"Train step: {self._train_step.numpy()}")
                print("=============================================")
                wandb.log(
                    {
                        "Epoch": self._total_episodes,
                        "Score": self._episode_reward,
                        "Steps": self._episode_steps,
                    },
                    step=self._train_step.numpy(),
                )

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs, _ = self._env.reset()

                # Load content of variables
                self._variable_container.update_variables()
            else:
                # Super critical !!!
                self._last_obs = new_obs

        # send all experiences to DB server
        writer.flush()

    def run(self):
        # Init environment
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._last_obs, _ = self._env.reset()

        # Connect to database
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            for _ in range(0, self._warmup_steps, self._env_steps):
                # Warmup steps
                self.collect(writer, self._env_steps, self.random_policy)

            # Main loop
            while not self._stop_agents:
                # Re-new noise matrix
                self.model.reset_noise()

                self.collect(writer, self._env_steps, self.collect_policy)

    def save(self, path=""):
        if self._save_path:
            try:
                os.makedirs(os.path.join(os.path.join(self._save_path, path)))
            except OSError:
                print("The path already exist ❗❗❗")
            finally:
                # Save model
                self.model.save_weights(
                    os.path.join(os.path.join(self._save_path, path), "actor.h5")
                )
                wandb.save(
                    os.path.join(os.path.join(self._save_path, path), "actor.h5")
                )
