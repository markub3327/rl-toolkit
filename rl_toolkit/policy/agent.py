from rl_toolkit.networks import Actor

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
        env_steps (int): maximum number of steps in each rollout
        learning_starts (int): number of interactions before using policy network
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        db_server: str,
        # ---
        env,
        env_steps: int = 64,
        learning_starts: int = 10000,
        # ---
        log_wandb: bool = False,
    ):
        self._env = env
        self._env_steps = env_steps
        self._learning_starts = learning_starts
        self._log_wandb = log_wandb

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

        # actual training step
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
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

        # prepare variable container
        self._variables_container = {
            "train_step": self._train_step,
            "stop_agents": self._stop_agents,
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

        self._policy_action = self._get_action(
            self._last_obs, deterministic=False
        ).numpy
        self._random_action = self._env.action_space.sample

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.env_steps = env_steps
            wandb.config.learning_starts = learning_starts

    def _normalize_fn(self, obs):
        # Min-max method
        return obs / self._env.observation_space.high

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

    def _log_train(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"Train step: {self._train_step}")
        print("=============================================")

        if self._log_wandb:
            wandb.log(
                {
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
                },
                # step=step,
            )

    def _collect_rollout(self, steps, action_fn, writer):
        for _ in range(steps):
            # Step in the environment
            action = action_fn()
            new_obs, reward, terminal, _ = self._env.step(action)
            new_obs = self._normalize(new_obs)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1

            # Update the replay buffer
            writer.append(
                {
                    "observation": self._last_obs,
                    "action": action,
                    "reward": np.array([reward], dtype=np.float32),
                    "terminal": np.array([terminal], dtype=np.float32),
                }
            )

            # Ak je v cyklickom bufferi dostatok prikladov
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

            # Check the end of episode
            if terminal:
                self._log_train()

                # Write the final state !!!
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

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs = self._env.reset()
                self._last_obs = self._normalize(self._last_obs)

                # Interrupt the rollout
                break

            # Super critical !!!
            self._last_obs = new_obs

        writer.flush()

    def run(self):
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()
        self._last_obs = self._normalize(self._last_obs)

        # spojenie s db
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            # zahrievacie kola
            self._collect_rollout(self._learning_starts, self._random_action, writer)

            # hlavny cyklus hry
            while not self._stop_agents:
                # Update agent network
                self._update_variables()

                # Re-new noise matrix before every rollouts
                self._actor.reset_noise()

                # Collect rollouts
                self._collect_rollout(self._env_steps, self._policy_action, writer)
