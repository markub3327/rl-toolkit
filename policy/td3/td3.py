from .network import Actor, Critic
from policy.off_policy import OffPolicy
from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from utils.lr_scheduler import Linear as LinearScheduler

import wandb
import tensorflow as tf


class TD3(OffPolicy):
    """
    Twin Delayed DDPG
    =================

    Paper: https://arxiv.org/pdf/1802.09477.pdf

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        env_steps (int): maximum number of steps in each rollout
        gradient_steps (int): number of update steps after each rollout
        learning_starts (int): number of interactions before using policy network
        buffer_size (int): the maximum size of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): learning rate for actor's optimizer
        critic_learning_rate (float): learning rate for critic's optimizer
        lr_scheduler (str): type of learning rate scheduler
        tau (float): the soft update coefficient for target networks
        gamma (float): the discount factor
        norm_obs (bool): normalize every observation
        noise_type (str): the type of noise generator
        action_noise (float): the scale of the action noise
        target_noise (float): the scale of the target noise
        noise_clip (float): the bound of target noise
        policy_delay (int): periodicity of updating policy and target networks
        model_a_path (str): path to the actor's model
        model_c1_path (str): path to the critic_1's model
        model_c2_path (str): path to the critic_2's model
        logging_wandb (bool): logging by WanDB
    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        env_steps: int = -1,
        gradient_steps: int = -1,
        # ---
        learning_starts: int = 10000,
        # ---
        buffer_size: int = 1000000,
        batch_size: int = 128,
        # ---
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        lr_scheduler: str = "none",
        # ---
        tau: float = 0.005,
        gamma: float = 0.99,
        norm_obs: bool = False,
        # ---
        noise_type: str = "normal",
        action_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        # ---
        model_a_path: str = None,
        model_c1_path: str = None,
        model_c2_path: str = None,
        logging_wandb: bool = False,
    ):
        super(TD3, self).__init__(
            env=env,
            max_steps=max_steps,
            env_steps=env_steps,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            norm_obs=norm_obs,
            logging_wandb=logging_wandb,
        )

        self._target_noise = tf.constant(target_noise)
        self._noise_clip = tf.constant(noise_clip)
        self._policy_delay = policy_delay
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate

        # select noise generator
        if noise_type == "normal":
            self._noise = NormalActionNoise(
                mean=0.0, sigma=action_noise, shape=self.model.output_shape[1:]
            )
        elif noise_type == "ornstein-uhlenbeck":
            self._noise = OrnsteinUhlenbeckActionNoise(
                mean=0.0, sigma=action_noise, shape=self.model.output_shape[1:]
            )
        else:
            raise NameError(f"'{noise_type}' noise is not defined")

        # init LR scheduler
        if lr_scheduler == "none":
            self._actor_learning_rate = actor_learning_rate
            self._critic_learning_rate = critic_learning_rate
        elif lr_scheduler == "linear":
            self._actor_learning_rate = LinearScheduler(
                initial_value=actor_learning_rate, max_step=max_steps, warmup_steps=learning_starts
            )
            self._critic_learning_rate = LinearScheduler(
                initial_value=critic_learning_rate, max_step=max_steps, warmup_steps=learning_starts
            )
        else:
            raise NameError(f"'{lr_scheduler}' learning rate scheduler is not defined")

        # logging metrics
        self._loss_a = tf.keras.metrics.Mean()
        self._loss_c1 = tf.keras.metrics.Mean()
        self._loss_c2 = tf.keras.metrics.Mean()

        # Actor network & target network
        self._actor = Actor(
            noise_type=noise_type,
            action_noise=action_noise,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._actor_learning_rate,
            model_path=model_a_path,
        )
        self._actor_targ = Actor(
            noise_type=noise_type,
            action_noise=action_noise,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._actor_learning_rate,
            model_path=model_a_path,
        )

        # Critic network & target network
        self._critic_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._critic_learning_rate,
            model_path=model_c1_path,
        )
        self._critic_targ_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._critic_learning_rate,
            model_path=model_c1_path,
        )

        # Critic network & target network
        self._critic_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._critic_learning_rate,
            model_path=model_c2_path,
        )
        self._critic_targ_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=self._critic_learning_rate,
            model_path=model_c2_path,
        )

        # first make a hard copy
        self._update_target(self._actor, self._actor_targ, tau=tf.constant(1.0))
        self._update_target(self._critic_1, self._critic_targ_1, tau=tf.constant(1.0))
        self._update_target(self._critic_2, self._critic_targ_2, tau=tf.constant(1.0))

        # init Weights & Biases
        if self._logging_wandb:
            wandb.init(project="rl-toolkit")
            ###
            ### Settings
            ###
            wandb.config.max_steps = max_steps
            wandb.config.env_steps = env_steps
            wandb.config.gradient_steps = gradient_steps
            wandb.config.learning_starts = learning_starts
            wandb.config.buffer_size = buffer_size
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.lr_scheduler = lr_scheduler
            wandb.config.tau = tau
            wandb.config.gamma = gamma
            wandb.config.norm_obs = norm_obs
            wandb.config.noise_type = noise_type
            wandb.config.action_noise = action_noise
            wandb.config.target_noise = target_noise
            wandb.config.noise_clip = noise_clip
            wandb.config.policy_delay = policy_delay

    @tf.function
    def _get_action(self, state, deterministic):
        a = self._actor.model(tf.expand_dims(state, axis=0))
        a = tf.squeeze(a, axis=0)  # remove batch_size dim
        if deterministic == False:
            a = tf.clip_by_value(a + self._noise.sample(), -1.0, 1.0)
        return a

    # ------------------------------------ update critic ----------------------------------- #
    @tf.function
    def _update_critic(self, batch):
        next_action = self._actor_targ.model(batch["obs2"])

        # target policy smoothing
        epsilon = tf.random.normal(
            next_action.shape, mean=0.0, stddev=self._target_noise
        )
        epsilon = tf.clip_by_value(epsilon, -self._noise_clip, self._noise_clip)
        next_action = tf.clip_by_value(next_action + epsilon, -1.0, 1.0)

        # target Q-values
        q_1 = self._critic_targ_1.model([batch["obs2"], next_action])
        q_2 = self._critic_targ_2.model([batch["obs2"], next_action])
        next_q = tf.minimum(q_1, q_2)

        # Use Bellman Equation
        Q_targets = batch["rew"] + (1 - batch["done"]) * self._gamma * next_q

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self._critic_1.model([batch["obs"], batch["act"]])
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=q_values
            )
            q1_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q1_loss, self._critic_1.model.trainable_variables)
        self._critic_1.optimizer.apply_gradients(
            zip(grads, self._critic_1.model.trainable_variables)
        )

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self._critic_2.model([batch["obs"], batch["act"]])
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=q_values
            )
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self._critic_2.model.trainable_variables)
        self._critic_2.optimizer.apply_gradients(
            zip(grads, self._critic_2.model.trainable_variables)
        )

        return q1_loss, q2_loss

    # ------------------------------------ update actor ----------------------------------- #
    @tf.function
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred = self._actor.model(batch["obs"])
            # predict q value
            q_pred = self._critic_1.model([batch["obs"], y_pred])

            # compute per example loss
            a_loss = tf.nn.compute_average_loss(-q_pred)

        grads = tape.gradient(a_loss, self._actor.model.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.model.trainable_variables)
        )

        return a_loss

    def _update(self):
        for gradient_step in tf.range(1, self._gradient_steps + 1):
            batch = self._rpm.sample(self._batch_size)

            # Critic models update
            l_c1, l_c2 = self._update_critic(batch)
            self._loss_c1.update_state(l_c1)
            self._loss_c2.update_state(l_c2)

            # Delayed policy update
            if gradient_step % self._policy_delay == 0:
                self._loss_a.update_state(self._update_actor(batch))

                # ---------------------------- soft update target networks ---------------------------- #
                self._update_target(self._actor, self._actor_targ, tau=self._tau)
                self._update_target(self._critic_1, self._critic_targ_1, tau=self._tau)
                self._update_target(self._critic_2, self._critic_targ_2, tau=self._tau)

            # print(gradient_step, self.loss_a.result(), self.loss_c1.result(), self.loss_c2.result())

    def _logging_models(self):
        if self._logging_wandb:
            # logging of epoch's mean loss
            wandb.log(
                {
                    "loss_a": self._loss_a.result(),
                    "loss_c1": self._loss_c1.result(),
                    "loss_c2": self._loss_c2.result(),
                },
                step=self._total_steps,
            )
        self._clear_metrics()  # clear stored metrics of losses

    def _clear_metrics(self):
        # reset logger
        self._loss_a.reset_states()
        self._loss_c1.reset_states()
        self._loss_c2.reset_states()

    def save(self, path):
        # Save model to local drive
        self._actor.model.save(f"{path}model_A.h5")
        self._critic_1.model.save(f"{path}model_C1.h5")
        self._critic_2.model.save(f"{path}model_C2.h5")
