from rl_toolkit.networks import Actor
from rl_toolkit.policy import Policy
from rl_toolkit.utils import VariableContainer

import reverb
import wandb

import numpy as np


class Agent(Policy):
    """
    Agent (based on Soft Actor-Critic)
    =================

    Attributes:
        db_server (str): database server name (IP or domain name)
        env: the instance of environment object
        update_interval (int): interval of updating policy parameters
        log_wandb (bool): log into WanDB cloud

    Paper: https://arxiv.org/pdf/1812.05905.pdf
    """

    def __init__(
        self,
        # ---
        db_server: str,
        # ---
        env,
        warmup_steps: int = 10000,
        # ---
        update_interval: int = 64,
        # ---
        log_wandb: bool = False,
    ):
        super(Agent, self).__init__(env, log_wandb)

        self._update_interval = update_interval
        self._warmup_steps = warmup_steps

        # Actor network (for agent)
        self._actor = Actor(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
        )
        self._container = VariableContainer(db_server, self._actor)

        # Initializes the reverb client
        self.client = reverb.Client(f"{db_server}:8000")

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.warmup_steps = warmup_steps
            wandb.config.update_interval = update_interval

    def run(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()
        self._last_obs = self._normalize(self._last_obs)

        # spojenie s db
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            # hlavny cyklus hry
            while not self._container.stop_agents:
                # Get the action
                if self._total_steps < self._warmup_steps:
                    action = self._env.action_space.sample()

                    self._container.train_step.assign(self._total_steps)
                else:
                    if (self._total_steps % self._update_interval) == 0:
                        # Update agent network
                        self._container.update_variables()
                        # Re-new noise matrix before every rollouts
                        self._actor.reset_noise()

                    action = self._actor.get_action(
                        self._last_obs, deterministic=False
                    ).numpy()

                # perform action
                new_obs, reward, terminal, _ = self._env.step(action)
                new_obs = self._normalize(new_obs)

                # Update variables
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
                    # logovanie do konzoly
                    print("=============================================")
                    print(f"Epoch: {self._total_episodes}")
                    print(f"Score: {self._episode_reward}")
                    print(f"Steps: {self._episode_steps}")
                    print(f"TotalInteractions: {self._total_steps}")
                    print("=============================================")

                    if self._log_wandb:
                        wandb.log(
                            {
                                "epoch": self._total_episodes,
                                "score": self._episode_reward,
                                "steps": self._episode_steps,
                            },
                            step=self._container.train_step.numpy(),
                        )

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

                    # write all trajectories to db
                    writer.end_episode()

                # Super critical !!!
                self._last_obs = new_obs
