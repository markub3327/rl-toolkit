from rl_toolkit.networks.layers import Actor, MultiCritic
from tensorflow.keras import Model

import tensorflow as tf


class ActorCritic(Model):
    """Combines the actor and critic into an end-to-end model for training."""

    def __init__(self, num_of_outputs: int, gamma: float, tau: float, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(tau)

        # init param 'alpha' - Lagrangian constraint
        self.log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self.alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self.target_entropy = tf.cast(-num_of_outputs, dtype=tf.float32)

        # Actor
        self.actor = Actor(num_of_outputs)

        # Critic
        self.critic = MultiCritic(2)

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            # Q-value
            Q_values, action, log_pi = self.call(data["observation"])

            # target Q-value
            next_Q_values, _, next_log_pi = self.call(data["next_observation"])

            # Update 'Alpha'
            self.alpha.assign(tf.exp(self.log_alpha))
            losses = -1.0 * (
                self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(losses)

            # Bellman Equation
            Q_target = tf.stop_gradient(
                data["reward"]
                + (1.0 - data["terminal"])
                * self.gamma
                * (tf.reduce_min(next_Q_values, axis=1) - self.alpha * next_log_pi)
            )

            # Update 'Critic'
            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_target[:, tf.newaxis, :],
                y_pred=self.critic([data["observation"], data["action"]]),
            )
            Q_loss = tf.nn.compute_average_loss(losses)

            # Update 'Actor'
            losses = self.alpha * log_pi - tf.reduce_min(Q_values, axis=1)
            actor_loss = tf.nn.compute_average_loss(losses)

        # Update 'Alpha'
        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        # Update 'Critic'
        gradients = tape.gradient(Q_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.critic.trainable_variables)
        )

        # Update 'Actor'
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables)
        )

        # Re-new noise matrix every update of 'log_std' params
        self.actor.reset_noise()

        return {
            "actor_loss": actor_loss,
            "critic_loss": Q_loss,
            "alpha_loss": alpha_loss,
        }

    def call(self, inputs):
        action, log_pi = self.actor(inputs, with_log_prob=True)
        Q_value = tf.reduce_min(self.critic([inputs, action]), axis=1)
        return [Q_value, action, log_pi]

    def compile(self, actor_optimizer, critic_optimizer, alpha_optimizer):
        super(ActorCritic, self).compile()
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
