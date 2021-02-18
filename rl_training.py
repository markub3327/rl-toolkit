import gym
import wandb
import numpy as np
import pybullet_envs

# policy
from policy import TD3, SAC

# utilities
from utils.replay_buffer import ReplayBuffer
from utils.lr_scheduler import linear


class RLTraining:
    """"""

    def __init__(
        self,
        env_name: str,
        max_steps: int,
        env_steps: int,
        gradient_steps: int,
        learning_starts: int,
        policy: str,
        update_after: int,
        replay_size: int,
        batch_size: int,
        learning_rate: float,
        lr_scheduler: str,
        tau: float,
        gamma: float,
        noise_type: str,
        action_noise: float,
        target_noise: float,
        noise_clip: float,
        policy_delay: int,
        model_a_path: str,
        model_c1_path: str,
        model_c2_path: str,
        logging_wandb: bool,
    ):
        self.max_steps = max_steps
        self.env_steps = env_steps
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.update_after = update_after
        self.logging_wandb = logging_wandb

        # Herne prostredie
        self._env = gym.make(env_name)

        # inicializuj prostredie Weights & Biases
        if self.logging_wandb:
            wandb.init(project="rl-toolkit")

            ###
            ### Settings
            ###
            wandb.config.gamma = gamma
            wandb.config.batch_size = batch_size
            wandb.config.tau = tau
            wandb.config.learning_rate = learning_rate
            wandb.config.lr_scheduler = lr_scheduler
            wandb.config.replay_size = replay_size
            wandb.config.learning_starts = learning_starts
            wandb.config.update_after = update_after
            wandb.config.gradient_steps = gradient_steps
            wandb.config.env_steps = env_steps
            wandb.config.max_steps = max_steps
            wandb.config.action_noise = action_noise
            wandb.config.target_noise = target_noise
            wandb.config.noise_clip = noise_clip
            wandb.config.policy_delay = policy_delay

        # select lr
        if lr_scheduler == "none":
            self.lr_scheduler = None
        elif lr_scheduler == "linear":
            self.lr_scheduler = linear
        else:
            raise NameError(f"'{lr_scheduler}' learning rate scheduler is not defined")

        # replay buffer
        self._rpm = ReplayBuffer(
            obs_dim=self._env.observation_space.shape,
            act_dim=self._env.action_space.shape,
            size=replay_size,
        )

        # init policy
        if policy == "td3":
            self._agent = TD3(
                state_shape=self._env.observation_space.shape,
                action_shape=self._env.action_space.shape,
                actor_learning_rate=learning_rate,
                critic_learning_rate=learning_rate,
                lr_scheduler=self.lr_scheduler,
                tau=tau,
                gamma=gamma,
                noise_type=noise_type,
                action_noise=action_noise,
                target_noise=target_noise,
                noise_clip=noise_clip,
                policy_delay=policy_delay,
                model_a_path=model_a_path,
                model_c1_path=model_c1_path,
                model_c2_path=model_c2_path,
            )
        elif policy == "sac":
            self._agent = SAC(
                state_shape=self._env.observation_space.shape,
                action_shape=self._env.action_space.shape,
                actor_learning_rate=learning_rate,
                critic_learning_rate=learning_rate,
                alpha_learning_rate=learning_rate,
                lr_scheduler=self.lr_scheduler,
                tau=tau,
                gamma=gamma,
                model_a_path=model_a_path,
                model_c1_path=model_c1_path,
                model_c2_path=model_c2_path,
            )
        else:
            raise NameError(f"Algorithm '{policy}' is not defined")

    def train(self):
        self._total_steps = 0
        self._norm_timesteps = 0
        self._total_episodes = 0
        self._episode_reward = 0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # hlavny cyklus hry
        while self._total_steps < self.max_steps:
            # collect rollouts
            self._collect_rollouts()

            # update _norm_timesteps
            self._norm_timesteps = float(self._total_steps) / float(self.max_steps)
            print(float(self._total_steps), float(self.max_steps))

            # update models
            if (
                self._total_steps >= self.update_after
                and len(self._rpm) >= self.batch_size
            ):
                self._agent.update(
                    self._rpm,
                    self._norm_timesteps,
                    self.batch_size,
                    self.gradient_steps,
                    logging_wandb=self.logging_wandb,
                )

    def save(self, save_path):
        # Save model to local drive
        self._agent.actor.model.save(f"{save_path}model_A.h5")
        self._agent.critic_1.model.save(f"{save_path}model_C1.h5")
        self._agent.critic_2.model.save(f"{save_path}model_C2.h5")

    def close(self):
        # exit environment
        self._env.close()

    def _logging(self):
        print("=============================================")
        print(f"Epoch: {self._total_episodes}")
        print(f"EpsReward: {self._episode_reward}")
        print(f"EpsSteps: {self._episode_steps}")
        print(f"TotalInteractions: {self._total_steps}")
        print(f"ReplayBuffer: {len(self._rpm)}")
        print("=============================================")
        print(f"Training ... {round(self._norm_timesteps * 100.0)} %")
        if self.logging_wandb:
            wandb.log(
                {
                    "epoch": self._total_episodes,
                    "score": self._episode_reward,
                    "steps": self._episode_steps,
                    "replayBuffer": len(self._rpm),
                }
            )

    def _collect_rollouts(self):
        # re-new noise matrix before every rollouts
        self._agent.actor.reset_noise()

        # collect rollouts
        for env_step in range(self.env_steps):
            # select action randomly or using policy network
            if self._total_steps < self.learning_starts:
                # warmup
                action = self._env.action_space.sample()
            else:
                # Get the noisy action
                action = self._agent.get_action(self._last_obs).numpy()

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
                self._logging()

                self._episode_reward = 0
                self._episode_steps = 0
                self._total_episodes += 1

                # init environment
                self._last_obs = self._env.reset()

                # interrupt the rollout
                break

            # super critical !!!
            self._last_obs = new_obs
