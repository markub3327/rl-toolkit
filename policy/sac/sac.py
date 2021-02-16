from policy.off_policy import OffPolicy
from .network import Actor, Critic

import wandb
import tensorflow as tf


class SAC(OffPolicy):
    """
    Soft Actor-Critic

    https://arxiv.org/pdf/1812.05905.pdf
    """

    def __init__(
        self,
        env,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        tau: float,
        gamma: float,
        env_steps: int,
        model_a_path: str,
        model_c1_path: str,
        model_c2_path: str,
    ):
        super(SAC, self).__init__(
            env=env,
            tau=tau,
            gamma=gamma,
        )

        self.env_steps = env_steps
        self.last_obs = None        # empty state

        # logging metrics
        self.loss_a = tf.keras.metrics.Mean()
        self.loss_c1 = tf.keras.metrics.Mean()
        self.loss_c2 = tf.keras.metrics.Mean()
        self.loss_alpha = tf.keras.metrics.Mean()

        # init param 'alpha' - Lagrangian
        self._log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self._alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self._alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate, name="alpha_optimizer"
        )
        self._target_entropy = tf.cast(-tf.reduce_prod(self.env.action_space.shape), dtype=tf.float32)
        # print(self._target_entropy)
        # print(self._alpha)

        # Actor network
        self.actor = Actor(
            state_shape=self.env.observation_space.shape, 
            action_shape=self.env.action_space.shape, 
            lr=actor_learning_rate, 
            model_path=model_a_path
        )

        # Critic network & target network
        self.critic_1 = Critic(
            state_shape=self.env.observation_space.shape, 
            action_shape=self.env.action_space.shape, 
            lr=critic_learning_rate, 
            model_path=model_c1_path
        )
        self.critic_targ_1 = Critic(
            state_shape=self.env.observation_space.shape, 
            action_shape=self.env.action_space.shape, 
            lr=critic_learning_rate, 
            model_path=model_c1_path
        )

        # Critic network & target network
        self.critic_2 = Critic(
            state_shape=self.env.observation_space.shape, 
            action_shape=self.env.action_space.shape, 
            lr=critic_learning_rate, 
            model_path=model_c2_path
        )
        self.critic_targ_2 = Critic(
            state_shape=self.env.observation_space.shape, 
            action_shape=self.env.action_space.shape, 
            lr=critic_learning_rate, 
            model_path=model_c2_path
        )

        # first make a hard copy
        self.update_target(self.critic_1, self.critic_targ_1, tau=tf.constant(1.0))
        self.update_target(self.critic_2, self.critic_targ_2, tau=tf.constant(1.0))

    @tf.function
    def get_action(self, state):
        a, _ = self.actor.predict(tf.expand_dims(state, axis=0), with_logprob=False)
        return tf.squeeze(a, axis=0)  # remove batch_size dim

    # ------------------------------------ update critic ----------------------------------- #
    @tf.function
    def _update_critic(self, batch):
        next_action, next_log_pi = self.actor.predict(batch["obs2"])

        # target Q-values
        next_q_1 = self.critic_targ_1.model([batch["obs2"], next_action])
        next_q_2 = self.critic_targ_2.model([batch["obs2"], next_action])
        next_q = tf.minimum(next_q_1, next_q_2)
        # tf.print(f'nextQ: {next_q.shape}')

        # Bellman Equation
        Q_targets = tf.stop_gradient(
            batch["rew"]
            + (1 - batch["done"]) * self.gamma * (next_q - self._alpha * next_log_pi)
        )
        # tf.print(f'qTarget: {Q_targets.shape}')

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self.critic_1.model([batch["obs"], batch["act"]])
            q_losses = tf.losses.mean_squared_error(
                y_true=Q_targets, y_pred=q_values
            )
            q1_loss = tf.nn.compute_average_loss(q_losses)
        #    tf.print(f'q_val: {q_values.shape}')

        grads = tape.gradient(q1_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(
            zip(grads, self.critic_1.model.trainable_variables)
        )

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self.critic_2.model([batch["obs"], batch["act"]])
            q_losses = tf.losses.mean_squared_error(
                y_true=Q_targets, y_pred=q_values
            )
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(
            zip(grads, self.critic_2.model.trainable_variables)
        )

        return q1_loss, q2_loss

    # ------------------------------------ update actor ----------------------------------- #
    @tf.function
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_pi = self.actor.predict(batch["obs"])
            # tf.print(f'log_pi: {log_pi.shape}')

            # predict q value
            q_1 = self.critic_1.model([batch["obs"], y_pred])
            q_2 = self.critic_2.model([batch["obs"], y_pred])
            q = tf.minimum(q_1, q_2)
            # tf.print(f'q: {q.shape}')

            a_losses = self._alpha * log_pi - q
            a_loss = tf.nn.compute_average_loss(a_losses)
            # tf.print(f'a_losses: {a_losses}')

        grads = tape.gradient(a_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))

        return a_loss

    # ------------------------------------ update alpha ----------------------------------- #
    @tf.function
    def _update_alpha(self, batch):
        y_pred, log_pi = self.actor.predict(batch["obs"])
        # tf.print(f'y_pred: {y_pred.shape}')
        # tf.print(f'log_pi: {log_pi.shape}')

        self._alpha.assign(tf.exp(self._log_alpha))
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._log_alpha * tf.stop_gradient(log_pi + self._target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        #    tf.print(f'alpha_losses: {alpha_losses.shape}')

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    def update(self, rpm, batch_size, gradient_steps, logging_wandb):
        for gradient_step in range(1, gradient_steps + 1):
            batch = rpm.sample(batch_size)

            # re-new noise matrix every update of 'log_std' params
            self.actor.reset_noise()

            # Alpha param update
            self.loss_alpha.update_state(self._update_alpha(batch))

            # Critic models update
            l_c1, l_c2 = self._update_critic(batch)
            self.loss_c1.update_state(l_c1)
            self.loss_c2.update_state(l_c2)

            # Actor model update
            self.loss_a.update_state(self._update_actor(batch))

            # ---------------------------- soft update target networks ---------------------------- #
            self.update_target(self.critic_1, self.critic_targ_1, tau=self.tau)
            self.update_target(self.critic_2, self.critic_targ_2, tau=self.tau)

            # print(gradient_step, self.loss_a.result(), self.loss_c1.result(), self.loss_c2.result(), self.loss_alpha.result())

        # logging of epoch's mean loss
        if logging_wandb:
            wandb.log(
                {
                    "loss_a": self.loss_a.result(),
                    "loss_c1": self.loss_c1.result(),
                    "loss_c2": self.loss_c2.result(),
                    "loss_alpha": self.loss_alpha.result(),
                    "alpha": self._alpha,
                }
            )

        # reset logger
        self.loss_a.reset_states()
        self.loss_c1.reset_states()
        self.loss_c2.reset_states()
        self.loss_alpha.reset_states()

    def run(self, rpm):
        env_reward, env_timesteps = 0.0, 0

        # reset noise
        self.actor.reset_noise()

        # collect rollouts
        for env_step in range(self.env_steps):
            # Get the noisy action
            action = self.get_action(self.last_obs).numpy()

            # Step in the environment
            new_obs, reward, done, _ = self.env.step(action)

            # update variables
            env_reward += reward
            env_timesteps += 1

            # Update the replay buffer
            rpm.store(self.last_obs, action, reward, new_obs, done)

            # check end of episode
            if done:
                break

            # super critical !!!
            self.last_obs = new_obs

        return env_reward, env_timesteps, done