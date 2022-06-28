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
        render (bool): enable the rendering into the video file
        db_server (str): database server name (IP or domain name)
        actor_units (list): list of the numbers of units in each Actor's layer
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        init_noise (float): initialization of the Actor's noise
        warmup_steps (int): number of interactions before using policy network
        env_steps (int): number of steps per rollout
    """

    def __init__(
        self,
        # ---
        env_name: str,
        render: bool,
        db_server: str,
        # ---
        actor_units: list,
        clip_mean_min: float,
        clip_mean_max: float,
        init_noise: float,
        # ---
        warmup_steps: int,
        env_steps: int,
    ):
        super(Agent, self).__init__(env_name, render)

        self._env_steps = env_steps
        self._warmup_steps = warmup_steps

        # Init actor's network
        self.actor = Actor(
            units=actor_units,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            init_noise=init_noise,
        )
        self.actor.build((None,) + self._env.observation_space.shape)

        # Show models details
        self.actor.summary()

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
            table="variable",
            variables={
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.actor.variables,
            },
        )

        # load content of variables & re-new noise matrix
        self._variable_container.update_variables()
        self.actor.reset_noise()

        # Initializes the reverb client
        self.client = reverb.Client(db_server)

        # init Weights & Biases
        wandb.init(
            project="rl-toolkit",
            group=f"{env_name}",
            monitor_gym=render,
        )
        wandb.config.warmup_steps = warmup_steps
        wandb.config.env_steps = env_steps

    def random_policy(self, input):
        action = self._env.action_space.sample()
        return action

    @tf.function(jit_compile=True)
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
            action = np.array(action, copy=False)

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # Update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Update the replay buffer
            writer.append(
                {
                    "observation": self._last_obs.astype("float32", copy=False),
                    "action": action,
                    "reward": np.array([reward], copy=False, dtype="float32"),
                    "terminal": np.array([terminal], copy=False),
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
                writer.append({"observation": new_obs.astype("float32", copy=False)})
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

                # Block until all the items have been sent to the server
                writer.end_episode()

                # logovanie
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
                self._last_obs = self._env.reset()
            else:
                # Super critical !!!
                self._last_obs = new_obs

        writer.flush()

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
