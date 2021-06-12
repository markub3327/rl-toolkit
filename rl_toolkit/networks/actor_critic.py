from rl_toolkit.networks.layers import Actor, Critic
from tensorflow.keras import Model

import tensorflow as tf


class ActorCritic(Model):
    """Combines the actor and critic into an end-to-end model for training."""

    def __init__(self, num_of_outputs: int, gamma: float, tau: float, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(tau)

        # Actor
        self.actor = Actor(num_of_outputs)

        # Critic 1
        self.critic_1 = Critic()
        self.critic_1_target = Critic()
        self._train_target(self.critic_1, self.critic_1_target, tau=tf.constant(1.0))

        # Critic 2
        self.critic_2 = Critic()
        self.critic_2_target = Critic()
        self._train_target(self.critic_2, self.critic_2_target, tau=tf.constant(1.0))

        # init param 'alpha' - Lagrangian constraint
        self.log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self.alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self.target_entropy = tf.cast(-num_of_outputs, dtype=tf.float32)

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            action, log_pi = self.actor(data["observation"], with_log_prob=True)
            next_action, next_log_pi = self.actor(
                data["next_observation"], with_log_prob=True
            )

            # update 'Alpha'
            self.alpha.assign(tf.exp(self.log_alpha))
            losses = -1.0 * (
                self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(losses)

            # target Q-values
            next_Q_values = tf.minimum(
                self.critic_1_target([data["next_observation"], next_action]),
                self.critic_2_target([data["next_observation"], next_action]),
            )

            # Bellman Equation
            Q_targets = tf.stop_gradient(
                data["reward"]
                + (1.0 - data["terminal"])
                * self.gamma
                * (next_Q_values - self.alpha * next_log_pi)
            )

            # update 'Critic 1'
            Q_values = self.critic_1([data["observation"], data["action"]])
            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=Q_values
            )
            Q1_loss = tf.nn.compute_average_loss(losses)

            # update 'Critic 2'
            Q_values = self.critic_2([data["observation"], data["action"]])
            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=Q_values
            )
            Q2_loss = tf.nn.compute_average_loss(losses)

            # update 'Actor'
            Q_values = tf.minimum(
                self.critic_1([data["observation"], action]),
                self.critic_2([data["observation"], action]),
            )
            losses = self.alpha * log_pi - Q_values
            actor_loss = tf.nn.compute_average_loss(losses)

        # Update parameters
        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        gradients = tape.gradient(Q1_loss, self.critic_1.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.critic_1.trainable_variables)
        )

        gradients = tape.gradient(Q2_loss, self.critic_2.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.critic_2.trainable_variables)
        )

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        # Soft update target networks
        self._train_target(self.critic_1, self.critic_1_target, tau=self.tau)
        self._train_target(self.critic_2, self.critic_2_target, tau=self.tau)

        # Re-new noise matrix every update of 'log_std' params
        self.actor.reset_noise()

        return {
            "critic_loss": (Q1_loss + Q2_loss),
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
        }

    def _train_target(self, source, target, tau):
        for source_weight, target_weight in zip(
            source.trainable_variables, target.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def call(self, inputs):
        pass
