from rl_toolkit.policy import Policy

import reverb
import wandb

import numpy as np


class Random(Policy):
    """
    Random agent
    =================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        model_a_path (str): path to the actor's model
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env,
        # ---
        db_server: str,
        # ---
        max_steps: int,
    ):
        super(Random, self).__init__(env, False)

        self._max_steps = max_steps

        # Initializes the reverb client
        self.client = reverb.Client(f"{db_server}:8000")

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
                step=self._total_steps,
            )

    def run(self):
        self._total_steps = 0
        self._episode_steps = 0

        # spojenie s db
        with self.client.trajectory_writer(num_keep_alive_refs=2) as writer:
            # hlavny cyklus hry
            while self._total_steps < self._max_steps:
                self._last_obs = self._env.reset()
                self._last_obs = self._normalize(self._last_obs)

                # collect rollout
                while True:
                    # Get the action
                    action = self._env.action_space.sample()

                    # perform action
                    new_obs, reward, terminal, _ = self._env.step(action)
                    new_obs = self._normalize(new_obs)

                    # update variables
                    self._episode_steps += 1
                    self._total_steps += 1

                    # super critical !!!
                    self._last_obs = new_obs

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
                        self._episode_steps = 0

                        # write all trajectories to db
                        writer.end_episode()

                        # Interrupt the rollout
                        break

                    # Super critical !!!
                    self._last_obs = new_obs
