from abc import ABC, abstractmethod

import wandb
import tensorflow as tf

# utilities
from utils.replay_buffer import ReplayBuffer
from utils.lr_scheduler import linear


class OffPolicy(ABC):
    """
    The base for Off-Policy algorithms
    :param tau: the soft update coefficient (float)
    :param gamma: the discount factor (float)
    :param lr_scheduler: type of learning rate scheduler
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
        update_after: int,
        # ---
        replay_size: int,
        batch_size: int,
        lr_scheduler: str,
        # ---
        tau: float,
        gamma: float,
        # ---
        logging_wandb: bool,
    ):
        self._env = env
        self._max_steps = max_steps
        self._env_steps = env_steps
        self._gradient_steps = gradient_steps
        self._learning_starts = learning_starts
        self._update_after = update_after
        self._batch_size = batch_size
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._logging_wandb = logging_wandb

        if lr_scheduler == "none":
            self._lr_scheduler = None
        elif lr_scheduler == "linear":
            self._lr_scheduler = linear
        else:
            raise NameError(f"'{lr_scheduler}' learning rate scheduler is not defined")

        # init replay buffer
        self._rpm = ReplayBuffer(
            obs_dim=self._env.observation_space.shape,
            act_dim=self._env.action_space.shape,
            size=replay_size,
        )

    @abstractmethod
    def _get_action(self, state, deterministic):
        ...

    @abstractmethod
    def _update_learning_rate(self, epoch):
        ...

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
            f"Training ... {round(float(self._total_steps) / float(self._max_steps) * 100.0)} %"
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
            f"Testing ... {round(float(self._total_steps) / float(self._max_steps) * 100.0)} %"
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

    def train(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # collect rollouts
            self._play()

            # update models
            if (
                self._total_steps >= self._update_after
                and len(self._rpm) >= self._batch_size
            ):
                self._update()
                self._logging_models()

    def test(self):
        self._total_steps = 0
        self._total_episodes = 0

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            self._episode_reward = 0.0
            self._episode_steps = 0
            done = False

            self._last_obs = self._env.reset()

            # collect rollout
            while not done:
                # env.render()

                # Get the action
                action = self._get_action(self._last_obs, deterministic=True).numpy()

                # perform action
                new_obs, reward, done, _ = self._env.step(action)

                # update variables
                self._episode_reward += reward
                self._episode_steps += 1
                self._total_steps += 1

                # super critical !!!
                self._last_obs = new_obs

            # increment episode
            self._total_episodes += 1

            # logovanie
            self._logging_test()

    # do a few environment steps
    def _play(self):
        # re-new noise matrix before every rollouts
        self._actor.reset_noise()

        # collect rollouts
        for env_step in range(self._env_steps):
            # select action randomly or using policy network
            if self._total_steps < self._learning_starts:
                # warmup
                action = self._env.action_space.sample()
            else:
                # Get the noisy action
                action = self._get_action(self._last_obs, deterministic=False).numpy()

            # Step in the environment
            new_obs, reward, done, _ = self._env.step(action)

            # update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Update the replay buffer
            self._rpm.store(self._last_obs, action, reward, new_obs, done)

            # check the end of episode
            if done:
                self._logging_train()

                self._episode_reward = 0
                self._episode_steps = 0
                self._total_episodes += 1

                # init environment
                self._last_obs = self._env.reset()

                # interrupt the rollout
                break

            # super critical !!!
            self._last_obs = new_obs

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
