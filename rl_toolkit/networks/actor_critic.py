from rl_toolkit.networks.layers import Actor, TwinCritic
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
        self.critic = TwinCritic()
        self.critic_target = TwinCritic()
        self._train_target(self.critic, self.critic_target, tau=tf.constant(1.0))

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

            # target Q-value
            next_Q1_value, next_Q2_value = self.critic_target(
                [data["next_observation"], next_action]
            )
            next_Q_value = tf.minimum(next_Q1_value, next_Q2_value)

            # Bellman Equation
            Q_target = tf.stop_gradient(
                data["reward"]
                + (1.0 - data["terminal"])
                * self.gamma
                * (next_Q_value - self.alpha * next_log_pi)
            )

            # update 'Critic'
            Q1_value, Q2_value = self.critic([data["observation"], data["action"]])
            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_target, y_pred=Q1_value
            )
            Q_loss = tf.nn.compute_average_loss(losses)
            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_target, y_pred=Q2_value
            )
            Q_loss += tf.nn.compute_average_loss(losses)

            # update 'Actor'
            Q1_value, Q2_value = self.critic([data["observation"], action])
            Q_value = tf.minimum(Q1_value, Q2_value)
            losses = self.alpha * log_pi - Q_value
            actor_loss = tf.nn.compute_average_loss(losses)

        # Update parameters
        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        gradients = tape.gradient(Q_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        # Soft update target networks
        self._train_target(self.critic, self.critic_target, tau=self.tau)

        # Re-new noise matrix every update of 'log_std' params
        self.actor.reset_noise()

        return {
            "alpha_loss": alpha_loss,
            "critic_loss": Q_loss,
            "actor_loss": actor_loss,
        }

    def _train_target(self, source, target, tau):
        for source_weight, target_weight in zip(
            source.trainable_variables, target.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def call(self, inputs):
        action, log_pi = self.actor(inputs, with_log_prob=True)

        Q1_value, Q2_value = self.critic([data["observation"], action])
        Q_value = tf.minimum(Q1_value, Q2_value)            
        
        Q1_value, Q2_value = self.critic([data["observation"], action])
        Q_value_target = tf.minimum(Q1_value, Q2_value)            
        
        return Q_value, Q_value_target, action, log_pi
