from rl_toolkit.networks import Actor

import math
import reverb
import wandb

import numpy as np
import tensorflow as tf


class Agent:
    """
    Agent (based on Soft Actor-Critic)
    =================

    Paper: https://arxiv.org/pdf/1812.05905.pdf

    Attributes:
        db_server (str): database server name (IP or domain name)
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        env_steps (int): maximum number of steps in each rollout
        learning_starts (int): number of interactions before using policy network
        logging_wandb (bool): logging by WanDB
    """

    def __init__(
        self,
        # ---
        db_server: str,
        # ---
        env,
        max_steps: int,
        env_steps: int = 64,
        learning_starts: int = 10000,
        # ---
        logging_wandb: bool = False,
    ):
        self._env = env
        self._max_steps = max_steps
        self._env_steps = env_steps
        self._learning_starts = learning_starts
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
        )

        # prepare variable container
        self._variables_container = {
            "policy_variables": self._actor.model.variables,
        }

        # variables signature for variable container table
        variable_container_signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self._variables_container,
        )
        self._dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, variable_container_signature
        )

        # Initializes the reverb client
        self.client = reverb.Client(f"{db_server}:8000")
        self.tf_client = reverb.TFClient(server_address=f"{db_server}:8000")

        # init Weights & Biases
        if self._logging_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.env_steps = env_steps
            wandb.config.learning_starts = learning_starts

    def _normalize_fn(self, obs):
        # Min-max method
        obs = (obs - self._env.observation_space.low) / (
            self._env.observation_space.high - self._env.observation_space.low
        )
        return obs

    @tf.function
    def _update_variables(self):
        sample = self.tf_client.sample("variables", data_dtypes=[self._dtypes])
        for variable, value in zip(
            tf.nest.flatten(self._variables_container), tf.nest.flatten(sample.data[0])
        ):
            variable.assign(value)

    @tf.function
    def _get_action(self, state, deterministic):
        a, _ = self._actor.predict(
            tf.expand_dims(state, axis=0),
            with_logprob=False,
            deterministic=deterministic,
        )
        return tf.squeeze(a, axis=0)  # remove batch_size dim

    def _logging_train(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print("=============================================")
        print(
            f"Playing ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
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

    def run(self):
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
                self._actor.reset_noise()

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
                    new_obs, reward, terminal, _ = self._env.step(action)
                    new_obs = self._normalize(new_obs)

                    # update variables
                    self._episode_reward += reward
                    self._episode_steps += 1
                    self._total_steps += 1

                    # Update the replay buffer
                    writer.append(
                        {
                            "observation": self._last_obs,
                            "action": action,
                            "reward": np.array([reward], dtype=np.float32),
                            "terminal": np.array([terminal], dtype=np.float32),
                        }
                    )

                    if self._episode_steps > 1:
                        writer.create_item(
                            table="experience",
                            priority=1.0,
                            trajectory={
                                "observation": writer.history["observation"][-2],
                                "action": writer.history["action"][-2],
                                "reward": writer.history["reward"][-2],
                                "next_observation": writer.history["observation"][-1],
                                "terminal": writer.history["terminal"][-2],
                            },
                        )

                    # check the end of episode
                    if terminal:
                        self._logging_train()

                        # write the final state !!!
                        writer.append({"observation": new_obs})
                        writer.create_item(
                            table="experience",
                            priority=1.0,
                            trajectory={
                                "observation": writer.history["observation"][-2],
                                "action": writer.history["action"][-2],
                                "reward": writer.history["reward"][-2],
                                "next_observation": writer.history["observation"][-1],
                                "terminal": writer.history["terminal"][-2],
                            },
                        )

                        # blocks until all the items have been sent to the server
                        # writer.end_episode(timeout_ms=1000)

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
