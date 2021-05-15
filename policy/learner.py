import reverb
import wandb

import tensorflow as tf


class Learner:
    """
    Agent
    =================

    Attributes:
        max_steps (int): maximum number of interactions do in environment

        buffer_size (int): maximum size of the replay buffer
        batch_size (int): size of mini-batch used for training

    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        # ---
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        # ---
        actor_learning_rate: float = 7.3e-4,
        critic_learning_rate: float = 7.3e-4,
        alpha_learning_rate: float = 7.3e-4,
        # ---
        learning_starts: int = int(1e4),
    ):
        self._max_steps = max_steps
        self._batch_size = batch_size
        self._env = env

        # logging metrics
        self._loss_a = tf.keras.metrics.Mean()
        self._loss_c1 = tf.keras.metrics.Mean()
        self._loss_c2 = tf.keras.metrics.Mean()
        self._loss_alpha = tf.keras.metrics.Mean()

        # Initialize the Reverb server
        self._db = reverb.Server(
            tables=[
                reverb.Table(
                    name="my_uniform_experience_replay_buffer",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(learning_starts),
                    # signature={
                    #    'actions': tf.TensorSpec(
                    #        [EPISODE_LENGTH, *ACTION_SPEC.shape],
                    #        ACTION_SPEC.dtype),
                    #    'observations': tf.TensorSpec(
                    #        [EPISODE_LENGTH + 1, *OBSERVATION_SPEC.shape],
                    #        OBSERVATION_SPEC.dtype),
                    # },
                ),
            ],
            port=8000
        )

        # ---------------- Init param 'alpha' (Lagrangian constraint) ---------------- #
        self._log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self._alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self._alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate, name="alpha_optimizer"
        )
        self._target_entropy = tf.cast(
            -tf.reduce_prod(self._env.action_space.shape), dtype=tf.float32
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit")

        # Settings
        wandb.config.max_steps = max_steps
        #wandb.config.env_steps = env_steps
        #wandb.config.gradient_steps = gradient_steps
        wandb.config.learning_starts = learning_starts
        wandb.config.buffer_size = buffer_size
        wandb.config.batch_size = batch_size
        wandb.config.actor_learning_rate = actor_learning_rate
        wandb.config.critic_learning_rate = critic_learning_rate
        wandb.config.alpha_learning_rate = alpha_learning_rate
        #wandb.config.tau = tau
        #wandb.config.gamma = gamma

    @tf.function
    def run(self):
        # hlavny cyklus ucenia
        for _ in range(self._max_steps):
            # get mini-batch from db
            batch = self._rpm.sample(self._batch_size)

            # re-new noise matrix every update of 'log_std' params
            self._actor.reset_noise()

            # Alpha param update
            self._loss_alpha.update_state(self._update_alpha(batch))

            l_c1, l_c2 = self._update_critic(batch)
            self._loss_c1.update_state(l_c1)
            self._loss_c2.update_state(l_c2)

            # Actor model update
            self._loss_a.update_state(self._update_actor(batch))

            # -------------------- soft update target networks -------------------- #
            self._update_target(self._critic_1, self._critic_targ_1, tau=self._tau)
            self._update_target(self._critic_2, self._critic_targ_2, tau=self._tau)

            # log to W&B
            wandb.log(
                {
                    "loss_a": self._loss_a.result(),
                    "loss_c1": self._loss_c1.result(),
                    "loss_c2": self._loss_c2.result(),
                    "loss_alpha": self._loss_alpha.result(),
                    "alpha": self._alpha,
                },
                step=self._total_steps,
            )

            # reset logger
            self._loss_a.reset_states()
            self._loss_c1.reset_states()
            self._loss_c2.reset_states()
            self._loss_alpha.reset_states()

    # -------------------------------- update alpha ------------------------------- #
    def _update_alpha(self, batch):
        _, log_pi = self._actor.predict(batch["obs"])

        self._alpha.assign(tf.exp(self._log_alpha))
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._log_alpha * tf.stop_gradient(log_pi + self._target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    # -------------------------------- update critic ------------------------------- #
    def _update_critic(self, batch):
        next_action, next_log_pi = self._actor.predict(batch["obs2"])

        # target Q-values
        next_q_1 = self._critic_targ_1.model([batch["obs2"], next_action])
        next_q_2 = self._critic_targ_2.model([batch["obs2"], next_action])
        next_q = tf.minimum(next_q_1, next_q_2)

        # Bellman Equation
        Q_targets = tf.stop_gradient(
            batch["rew"]
            + (1 - batch["done"]) * self._gamma * (next_q - self._alpha * next_log_pi)
        )

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

    # -------------------------------- update actor ------------------------------- #
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_pi = self._actor.predict(batch["obs"])

            # predict q value
            q_1 = self._critic_1.model([batch["obs"], y_pred])
            q_2 = self._critic_2.model([batch["obs"], y_pred])
            q = tf.minimum(q_1, q_2)

            a_losses = self._alpha * log_pi - q
            a_loss = tf.nn.compute_average_loss(a_losses)

        grads = tape.gradient(a_loss, self._actor.model.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.model.trainable_variables)
        )

        return a_loss
