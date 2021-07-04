import tensorflow as tf
from tensorflow.keras import Model

from rl_toolkit.networks.layers import Actor, MultiCritic


class ActorCritic(Model):
    """
    Actor-Critic
    ===========

    Attributes:
        num_of_outputs (int): number of outputs
        gamma (float): the discount factor

    References:
        - [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
    """

    def __init__(self, num_of_outputs: int, gamma: float, init_alpha: float, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)

        # init param 'alpha' - Lagrangian constraint
        self.log_alpha = tf.Variable(
            tf.math.log(init_alpha), trainable=True, name="log_alpha"
        )
        self.alpha = tf.Variable(init_alpha, trainable=False, name="alpha")
        self.target_entropy = tf.cast(-num_of_outputs, dtype=tf.float32)

        # Actor
        self.actor = Actor(num_of_outputs)

        # Critic
        self.critic = MultiCritic(2)

    def train_step(self, data):
        # Re-new noise matrix every update of 'log_std' params
        self.actor.reset_noise()

        # Update 'Alpha'
        self.alpha.assign(tf.exp(self.log_alpha))
        with tf.GradientTape() as tape:
            _, log_pi = self.actor(
                data["observation"],
                training=True,
                with_log_prob=True,
                deterministic=False,
            )

            losses = -1.0 * (
                self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(losses)

        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        # Update 'Critic'
        with tf.GradientTape() as tape:
            # concatenating the two batches
            merged_observation = tf.concat([data["observation"], data["next_observation"]], axis=0)
            next_action, next_log_pi = self.actor(
                data["next_observation"],
                training=True,
                with_log_prob=True,
                deterministic=False,
            )
            merged_action = tf.concat([data["action"], next_action], axis=0)

            # get Q-value
            Q_value, next_Q_value = tf.split(
                self.critic([merged_observation, merged_action], training=True), 
                num_or_size_splits=2,
                axis=0
            )
            next_Q_value = tf.reduce_min(
                next_Q_value, axis=1
            )
            tf.print(Q_value.shape)
            tf.print(next_Q_value.shape)

            # Bellman Equation
            Q_target = tf.stop_gradient(
                data["reward"]
                + (1.0 - data["terminal"])
                * self.gamma
                * (next_Q_value - self.alpha * next_log_pi)
            )

            losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_target[:, tf.newaxis, :],
                y_pred=Q_value
            )
            critic_loss = tf.nn.compute_average_loss(losses)

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.critic.trainable_variables)
        )

        # Update 'Actor'
        with tf.GradientTape() as tape:
            # Q-value
            Q_value, log_pi = self(data["observation"], training=True)

            # Update 'Actor'
            losses = self.alpha * log_pi - Q_value
            actor_loss = tf.nn.compute_average_loss(losses)

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables)
        )

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
        }

    def call(self, inputs, training=None):
        action, log_pi = self.actor(
            inputs, training=training, with_log_prob=True, deterministic=False
        )
        Q_value = tf.reduce_min(
            self.critic([inputs, action], training=False), axis=1
        )
        return [Q_value, log_pi]

    def compile(self, actor_optimizer, critic_optimizer, alpha_optimizer):
        super(ActorCritic, self).compile()
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
