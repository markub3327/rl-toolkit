from policy.off_policy import OffPolicy
from .network import Actor, Critic

import wandb
import tensorflow as tf


class TD3(OffPolicy):
    """
    Twin Delayed DDPG
    :param state_shape: the shape of state space
    :param action_shape: the shape of action space
    :param actor_learning_rate: learning rate for actor's optimizer (float)
    :param critic_learning_rate: learning rate for critic's optimizer (float)
    :param lr_scheduler: type of learning rate scheduler
    :param tau: the soft update coefficient for target networks (float)
    :param gamma: the discount factor (float)
    :param noise_type: the type of noise generator (str)
    :param action_noise: the scale of the action noise (float)
    :param target_noise: the scale of the target noise (float)
    :param noise_clip: the bound of target noise (float)
    :param policy_delay: periodicity of updating policy and target networks (int)
    :param model_a_path: path to the actor's model (str)
    :param model_c1_path: path to the critic_1's model (str)
    :param model_c2_path: path to the critic_2's model (str)

    https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(
        self,
        env,
        #---
        max_steps: int,
        env_steps: int,
        gradient_steps: int,
        #---
        learning_starts: int,
        update_after: int,
        #---
        replay_size: int,
        batch_size: int,
        #---
        actor_learning_rate: float,
        critic_learning_rate: float,
        lr_scheduler: str,
        #---
        tau: float, 
        gamma: float,
        #---
        noise_type: str,
        action_noise: float,
        target_noise: float,
        noise_clip: float,
        policy_delay: int,
        #---
        model_a_path: str,
        model_c1_path: str,
        model_c2_path: str,
        logging_wandb: bool
    ):
        super(TD3, self).__init__(
            env=env,
            max_steps=max_steps,
            env_steps=env_steps,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            update_after=update_after,
            replay_size=replay_size,
            batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            tau=tau,
            gamma=gamma,
            logging_wandb=logging_wandb
        )

        self._target_noise = tf.constant(target_noise)
        self._noise_clip = tf.constant(noise_clip)
        self._policy_delay = policy_delay
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate

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
            lr=actor_learning_rate,
            model_path=model_a_path,
        )
        self._actor_targ = Actor(
            noise_type=noise_type,
            action_noise=action_noise,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            lr=actor_learning_rate,
            model_path=model_a_path,
        )

        # Critic network & target network
        self._critic_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            lr=critic_learning_rate,
            model_path=model_c1_path,
        )
        self._critic_targ_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            lr=critic_learning_rate,
            model_path=model_c1_path,
        )

        # Critic network & target network
        self._critic_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            lr=critic_learning_rate,
            model_path=model_c2_path,
        )
        self._critic_targ_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            lr=critic_learning_rate,
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
            wandb.config.update_after = update_after
            wandb.config.replay_size = replay_size
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.lr_scheduler = lr_scheduler
            wandb.config.tau = tau
            wandb.config.gamma = gamma
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
            a = tf.clip_by_value(a + self._actor.noise.sample(), -1.0, 1.0)
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

    # ------------------------------------ update learning rate ----------------------------------- #
    def _update_learning_rate(self, epoch):
        tf.keras.backend.set_value(
            self._critic_1.optimizer.learning_rate,
            self._lr_scheduler(epoch, self._critic_learning_rate),
        )
        tf.keras.backend.set_value(
            self._critic_2.optimizer.learning_rate,
            self._lr_scheduler(epoch, self._critic_learning_rate),
        )
        tf.keras.backend.set_value(
            self._actor.optimizer.learning_rate,
            self._lr_scheduler(epoch, self._actor_learning_rate),
        )

    def _update(self):
        # Update learning rate by lr_scheduler
        if self._lr_scheduler is not None:
            self._update_learning_rate(float(self._total_steps) / float(self._max_steps))

        for gradient_step in range(1, self._gradient_steps + 1):
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
                    "critic_learning_rate": self._critic_1.optimizer.learning_rate,
                    "actor_learning_rate": self._actor.optimizer.learning_rate,
                },
                step=self._total_steps
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
