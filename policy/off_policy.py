from abc import ABC, abstractmethod

import math
import wandb
import tensorflow as tf
import numpy as np

# utilities
from utils.replay_buffer import ReplayBuffer


class OffPolicy(ABC):
    """
    The base for Off-Policy algorithms
    ==================================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        env_steps (int): maximum number of steps in each rollout
        gradient_steps (int): number of update steps after each rollout
        learning_starts (int): number of interactions before using policy network
        buffer_size (int): the maximum size of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        lr_scheduler (str): type of learning rate scheduler
        tau (float): the soft update coefficient for target networks
        gamma (float): the discount factor
        norm_obs (bool): normalize observation
        logging_wandb (bool): logging by WanDB
    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        env_steps: int,
        gradient_steps: int,
        # ---
        learning_starts: int,
        # ---
        buffer_size: int,
        batch_size: int,
        # ---
        tau: float,
        gamma: float,
        norm_obs: bool,
        # ---
        logging_wandb: bool,
    ):
        self._env = env
        self._max_steps = max_steps
        self._env_steps = tf.constant(env_steps)
        self._gradient_steps = tf.constant(gradient_steps)
        self._learning_starts = tf.constant(learning_starts)
        self._batch_size = tf.constant(batch_size)
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._logging_wandb = logging_wandb
        self._norm_obs = tf.constant(norm_obs)

        # init replay buffer
        self._rpm = ReplayBuffer(
            obs_dim=self._env.observation_space.shape,
            act_dim=self._env.action_space.shape,
            size=buffer_size,
        )

        # init
        self._last_obs = tf.Variable(tf.zeros(self._env.observation_space.shape), trainable=False, name="last_obs")
        self._total_steps = tf.Variable(0, dtype=tf.int32, trainable=False, name="total_steps")
        self._total_episodes = tf.Variable(0, dtype=tf.int32, trainable=False, name="total_episodes")
        self._episode_reward = tf.Variable(0.0, trainable=False, name="episode_reward")
        self._episode_steps = tf.Variable(0, dtype=tf.int32, trainable=False, name="episode_steps")

    @abstractmethod
    def _get_action(self, state, deterministic):
        ...

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    @abstractmethod
    def _update(self):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def _logging_models(self):
        ...

    def _logging_train(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print(f"ReplayBuffer: {len(self._rpm)}")
        print("=============================================")
        print(
            f"Training ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
        )
        if self._logging_wandb:
            wandb.log(
                {
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
                    "replayBuffer": len(self._rpm),
                },
                step=self._total_steps,
            )

    def _logging_test(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"Score: {self._episode_reward}")
        print(f"Steps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print("=============================================")
        print(
            f"Testing ... {math.floor(self._total_steps * 100.0 / self._max_steps)} %"
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

    def _normalize_obs(self, obs):
        if self._norm_obs:
            return (obs - self._env.observation_space.low) / (self._env.observation_space.high - self._env.observation_space.low)
        else:
            return obs

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    def _env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""
        state, reward, done, _ = self._env.step(action)
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.bool))

    def _tf_env_step(self, action: tf.Tensor):
        return tf.numpy_function(self._env_step, [action], [tf.float32, tf.float32, tf.bool])

    @tf.function
    def _collect_rollout(self):
        # re-new noise matrix before every rollouts
        self._actor.reset_noise()

        # collect rollouts
        for _ in tf.range(self._env_steps):
            # normalize
            self._normalize_obs(self._last_obs)
            print(self._last_obs)

            # select action randomly or using policy network
            if self._total_steps < self._learning_starts:
                # warmup
                action = self._env.action_space.sample()
            else:
                # Get the noisy action
                action = self._get_action(
                    self._last_obs, deterministic=False
                )

            # Step in the environment
            new_obs, reward, done = self._tf_env_step(action)

            # update variables
            self._episode_reward.assign_add(reward)
            self._episode_steps.assign_add(1)
            self._total_steps.assign_add(1)

            # Update the replay buffer
            #self._rpm.store(self._last_obs, action, reward, new_obs, done)

            # check the end of episode
            if done:
                # interrupt the rollout
                break

            # super critical !!!
            self._last_obs.assign(new_obs)

        return done

    def train(self):

        # init environment
        self._last_obs.assign(self._env.reset())

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # run agent
            done = self._collect_rollout()

            # check the end of episode
            if done:
                self._logging_train()

                self._episode_reward.assign(0.0)
                self._episode_steps.assign(0)
                self._total_episodes.assign_add(1)

                # init environment
                self._last_obs.assign(self._env.reset())

                # interrupt the rollout
                break


            # update models
            if (
                self._total_steps >= self._learning_starts
                and len(self._rpm) >= self._batch_size
            ):
                self._update()
                self._logging_models()

#    def test(self):
#        self._total_steps = 0
#        self._total_episodes = 0
#        obs_log, act_log, rew_log = [], [], []

        # hlavny cyklus hry
#        while self._total_steps < self._max_steps:
#            self._episode_reward = 0.0
#            self._episode_steps = 0
#            done = False

#            self._last_obs.assign(self._env.reset())

            # collect rollout
#            while not done:
                # normalize
#                self._normalize_obs(self._last_obs)

                # Get the action
#                action = self._get_action(self._last_obs, deterministic=True).numpy()

                # perform action
#                new_obs, reward, done, _ = self._env.step(action)

                # log
#                obs_log.append(self._last_obs)
#                act_log.append(action)
#                rew_log.append(reward)

                # update variables
#                self._episode_reward += reward
#                self._episode_steps += 1
#                self._total_steps += 1

                # super critical !!!
#                self._last_obs.assign(new_obs)

            # increment episode
#            self._total_episodes += 1

            # logovanie
#            self._logging_test()

        # convert to numpy
#        obs_log = np.array(obs_log)
#        act_log = np.array(act_log)
#        rew_log = np.array(rew_log)

        # save to csv
#        np.savetxt('obs_log.csv', obs_log, delimiter=';')
#        np.savetxt('act_log.csv', act_log, delimiter=';')
#       np.savetxt('rew_log.csv', rew_log, delimiter=';')
