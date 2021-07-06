import numpy as np
import reverb
import tensorflow as tf
import wandb

from rl_toolkit.networks.models import Actor
from rl_toolkit.utils import VariableContainer

from .policy import Policy


class Agent(Policy):
    """
    Agent
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
        warmup_steps (int): number of interactions before using policy network
        env_steps (int): number of steps per rollout
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        warmup_steps: int = 10000,
        env_steps: int = 64,
        # ---
        log_wandb: bool = False,
    ):
        super(Agent, self).__init__(env_name)

        self._env_steps = env_steps
        self._warmup_steps = warmup_steps
        self._log_wandb = log_wandb

        # Init actor's network
        self.actor = Actor(
            n_outputs=tf.reduce_prod(self._env.action_space.shape).numpy()
        )
        self.actor.build((None,) + self._env.observation_space.shape)

        # Show models details
        self.actor.summary()

        # Variables
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
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
            db_server,
            "variables",
            {
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.actor.variables,
            },
        )

        # load content of variables & re-new noise matrix
        self._variable_container.update_variables()
        self.actor.reset_noise()

        # Initializes the reverb client
        self.client = reverb.Client(f"{db_server}:8000")

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.warmup_steps = warmup_steps
            wandb.config.env_steps = env_steps

    def random_policy(self, input):
        action = self._env.action_space.sample()
        return action

    @tf.function
    def collect_policy(self, input):
        action, _ = self.actor(
            tf.expand_dims(input, axis=0),
            with_log_prob=False,
            deterministic=False,
        )
        return tf.squeeze(action, axis=0)

    def collect(self, writer, max_steps, policy):
        # collect the rollout
        for _ in range(max_steps):
            # Get the action
            action = policy(self._last_obs)
            action = np.array(action, dtype="float32")

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Update the replay buffer
            writer.append(
                {
                    "observation": self._last_obs.astype("float32"),
                    "action": action,
                    "reward": np.array([reward], dtype="float32"),
                    "terminal": np.array([terminal], dtype="float32"),
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
                # Write the final interaction !!!
                writer.append({"observation": new_obs.astype("float32")})
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

                # Block until the item has been inserted and confirmed by the server
                writer.flush()

                # logovanie
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print(f"Train step: {self._train_step.numpy()}")
                print("=============================================")
                if self._log_wandb:
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
                self._last_obs = self._env.reset()
            else:
                # Super critical !!!
                self._last_obs = new_obs

    def run(self):
        # init environment
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._last_obs = self._env.reset()

        # spojenie s db
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            # zahrievacie kola
            self.collect(writer, self._warmup_steps, self.random_policy)

            # hlavny cyklus hry
            while not self._stop_agents:
                self.collect(writer, self._env_steps, self.collect_policy)

                # load content of variables & re-new noise matrix
                self._variable_container.update_variables()
                self.actor.reset_noise()
