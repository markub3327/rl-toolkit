import os

import numpy as np
import reverb
import tensorflow as tf
import wandb

from rl_toolkit.networks.models import DuelingDQN
from rl_toolkit.utils import VariableContainer

from ...core.process import Process


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
        save_path (str): path to the models for saving
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        num_layers: int,
        embed_dim: int,
        ff_mult: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        gamma: float,
        tau: float,
        frame_stack: int,
        # ---
        temp_init: float,
        temp_min: float,
        temp_decay: float,
        warmup_steps: int,
        # ---
        save_path: str,
    ):
        super(Agent, self).__init__(env_name, False, frame_stack)

        self._warmup_steps = warmup_steps
        self._save_path = save_path
        self._temp_min = temp_min
        self._temp_decay = temp_decay
        self._temp_init = temp_init
        self._frame_stack = frame_stack

        if (
            self._env.unwrapped.spec is not None
            and self._env.unwrapped.spec.id == "HumanoidRobot-v0"
        ):
            self._env.connect()

        # Init actor's network
        self.model = DuelingDQN(
            self._env.action_space.n,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ff_mult=ff_mult,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            gamma=gamma,
            tau=tau,
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

    def random_policy(self, inputs, temp):
        action = self._env.action_space.sample()
        return action

    # @tf.function(jit_compile=True)
    def collect_policy(self, inputs, temp):
        return self.model.get_action(tf.expand_dims(inputs, axis=0), temp)

    # Collect the rollout
    def collect(self, writer, policy):
        # Get the action
        action = policy(self._last_obs, self._temp)
        action = np.array(action, copy=False, dtype=self._env.action_space.dtype)

        # Perform action
        new_obs, ext_reward, terminated, truncated, _ = self._env.step(action)

        # Update variables
        self._episode_reward += ext_reward
        self._episode_steps += 1
        self._total_steps += 1

        # decrement temperature
        self._temp *= self._temp_decay
        self._temp = max(self._temp_min, self._temp)

        print(self._last_obs)

        # Update the replay buffer
        writer.append(
            {
                "observation": self._last_obs[-1],
                "action": action,
                "ext_reward": np.array([ext_reward], dtype=np.float64),
                "terminal": np.array([terminated]),
            }
        )

        # Enough samples to store in the database
        if self._episode_steps > self._frame_stack:
            writer.create_item(
                table="experience",
                priority=1.0,
                trajectory={
                    "observation": writer.history["observation"][:-1],
                    "action": writer.history["action"][-2],
                    "ext_reward": writer.history["ext_reward"][-2],
                    "next_observation": writer.history["observation"][-self._frame_stack:],
                    "terminal": writer.history["terminal"][-2],
                },
            )

        # Check the end of episode
        if terminated or truncated:
            # Write the final interaction !!!
            writer.append(
                {
                    "observation": new_obs[-1],
                }
            )
            writer.create_item(
                table="experience",
                priority=1.0,
                trajectory={
                    "observation": writer.history["observation"][:-1],
                    "action": writer.history["action"][-2],
                    "ext_reward": writer.history["ext_reward"][-2],
                    "next_observation": writer.history["observation"][-self._frame_stack:],
                    "terminal": writer.history["terminal"][-2],
                },
            )

            # Block until all the items have been sent to the server
            writer.end_episode()

            # save the checkpoint
            if self._total_episodes > 0:
                if self._episode_reward > self._best_episode_reward:
                    self._best_episode_reward = self._episode_reward
                    self.save()
                    print(
                        f"Model is saved at {self._total_episodes} episode with score {self._best_episode_reward}"
                    )
                    wandb.log({"best_score": self._best_episode_reward}, commit=False)
            else:
                self._best_episode_reward = self._episode_reward

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
                    "Temperature": self._temp,
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
        self._temp = self._temp_init
        self._last_obs, _ = self._env.reset()

        # Connect to database
        with self.client.trajectory_writer(num_keep_alive_refs=(self._frame_stack + 1)) as writer:
            for _ in range(0, self._warmup_steps):
                # Warmup steps
                self.collect(writer, self.random_policy)

            # Main loop
            while not self._stop_agents:
                self.collect(writer, self.collect_policy)

    def save(self, path=""):
        if self._save_path:
            try:
                os.makedirs(os.path.join(os.path.join(self._save_path, path)))
            except OSError:
                print("The path already exist ❗❗❗")
            finally:
                # Save model
                self.model.save_weights(
                    os.path.join(
                        os.path.join(self._save_path, path),
                        f"dqn_{self._total_episodes}.h5",
                    )
                )
