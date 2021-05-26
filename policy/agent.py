import reverb
import wandb

import tensorflow as tf
import numpy as np

from .network import Actor
from .reverb_utils import ReverbPolicyContainer


class Agent:
    """
    Agent
    =================

    Attributes:
        env: the instance of environment object
    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        env_steps: int = 64,
        # ---
        learning_starts: int = int(1e4),
    ):
        self._env = env
        self._max_steps = max_steps
        self._env_steps = env_steps
        self._learning_starts = learning_starts

        # Init Actor's network
        self._actor = Actor(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
        )

        # Initializes the Reverb client
        self._db_client = reverb.Client("192.168.1.38:8000")
        self._reverb_policy_container = ReverbPolicyContainer("192.168.1.38", self._actor.model)

        # init Weights & Biases
        wandb.init(project="rl-toolkit")

        # Settings
        wandb.config.max_steps = self._max_steps
        wandb.config.env_steps = self._env_steps
        wandb.config.learning_starts = self._learning_starts

    def run(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # Sync actor's params with db
            self._reverb_policy_container.update()

            # re-new noise matrix before every rollouts
            self._actor.reset_noise()

            # init writer
            with self._db_client.trajectory_writer(num_keep_alive_refs=2) as writer:
                # collect rollouts
                for step in range(self._env_steps):
                    # select action randomly or using policy network
                    if self._total_steps < self._learning_starts:
                        # warmup
                        action = self._env.action_space.sample()
                    else:
                        action = self._get_action(self._last_obs, deterministic=False)

                    # Step in the environment
                    obs2, reward, terminal, _ = self._env.step(action)

                    # update variables
                    self._episode_reward += reward
                    self._episode_steps += 1
                    self._total_steps += 1

                    # Update the replay buffer
                    writer.append(
                        {
                            "obs": np.array(self._last_obs, dtype=np.float32),
                            "action": np.array(action, dtype=np.float32),
                            "reward": np.array([reward], dtype=np.float32),
                            "terminal": np.array([terminal], dtype=np.float32),
                        }
                    )

                    if step >= 1:
                        # Create an item referencing all the data.
                        writer.create_item(
                            table="uniform_table",
                            priority=1.0,
                            trajectory={
                                "obs": writer.history["obs"][-2],
                                "action": writer.history["action"][-2],
                                "reward": writer.history["reward"][-2],
                                "obs2": writer.history["obs"][-1],
                                "terminal": writer.history["terminal"][-2],
                            },
                        )

                    # check the end of episode
                    if terminal:
                        print("=============================================")
                        print(f"Epoch: {self._total_episodes}")
                        print(f"Score: {self._episode_reward}")
                        print(f"Steps: {self._episode_steps}")
                        print(f"TotalInteractions: {self._total_steps}")
                        print("=============================================")
                        print(
                            f"Running ... {(self._total_steps*100)/self._max_steps} %"
                        )

                        wandb.log(
                            {
                                "epoch": self._total_episodes,
                                "score": self._episode_reward,
                                "steps": self._episode_steps,
                                #        "replayBuffer": len(self._rpm),
                            },
                            step=self._total_steps,
                        )

                        self._episode_reward = 0.0
                        self._episode_steps = 0
                        self._total_episodes += 1

                        # init environment
                        self._last_obs = self._env.reset()

                        # interrupt the rollout
                        break

                    # super critical !!!
                    self._last_obs = obs2

                # Block until the item has been inserted and confirmed by the server.
                writer.flush()

    @tf.function
    def _get_action(self, state, deterministic):
        a, _ = self._actor.predict(
            tf.expand_dims(state, axis=0),
            with_logprob=False,
            deterministic=deterministic,
        )
        return tf.squeeze(a, axis=0)  # remove batch_size dim
