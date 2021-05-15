import reverb
import wandb

import tensorflow as tf


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
    ):
        self._env = env

        # Initializes the Reverb client
        self._db = reverb.Client("localhost:8000")
        print(self._db.server_info())

    @tf.function
    def run(self, max_steps):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()
        self._last_obs = self._normalize(self._last_obs)

        # hlavny cyklus hry
        while self._total_steps < max_steps:
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
                    action, _ = self._actor.predict(
                        tf.expand_dims(self._last_obs, axis=0),
                        with_logprob=False,
                        deterministic=False,
                    )

                # Step in the environment
                new_obs, reward, done = self._tf_env_step(action)
                new_obs = self._normalize(new_obs)

                # update variables
                self._episode_reward += reward
                self._episode_steps += 1
                self._total_steps += 1

                # Update the replay buffer
                self._db.store(self._last_obs, action, reward, new_obs, done)

                # check the end of episode
                if done:
                    print("=============================================")
                    print(f"Epoch: {self._total_episodes}")
                    print(f"Score: {self._episode_reward}")
                    print(f"Steps: {self._episode_steps}")
                    print(f"TotalInteractions: {self._total_steps}")
                    print(f"ReplayBuffer: {len(self._rpm)}")
                    print("=============================================")
                    print(
                        f"Training ... {self._total_steps * 100.0 / self._max_steps} %"
                    )

                    wandb.log(
                        {
                            "epoch": self._total_episodes,
                            "score": self._episode_reward,
                            "steps": self._episode_steps,
                            "replayBuffer": len(self._rpm),
                        },
                        step=self._total_steps,
                    )

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

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    def _env_step(self, action):
        state, reward, done, _ = self._env.step(action)
        return (state, reward, done)

    def _tf_env_step(self, action):
        return tf.numpy_function(
            self._env_step, [action], [tf.float32, tf.float32, tf.float32]
        )
