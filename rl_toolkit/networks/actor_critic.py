import tensorflow as tf
from tensorflow.keras import Model

from rl_toolkit.networks.models import Actor, MultiCritic


class ActorCritic(Model):
    """
    Actor-Critic
    ===========

    Attributes:
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        n_outputs (int): number of outputs
        gamma (float): the discount factor
        init_alpha (float): initialization of alpha param

    References:
        - [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
    """

    def __init__(
        self,
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        n_outputs: int,
        gamma: float,
        init_alpha: float,
        **kwargs,
    ):
        super(ActorCritic, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(
            (tf.range(n_quantiles, dtype=tf.float32) + 0.5) / n_quantiles
        )

        # init param 'alpha' - Lagrangian constraint
        self.log_alpha = tf.Variable(
            tf.math.log(init_alpha), trainable=True, name="log_alpha"
        )
        self.alpha = tf.Variable(init_alpha, trainable=False, name="alpha")
        self.target_entropy = tf.cast(-n_outputs, dtype=tf.float32)

        # Actor
        self.actor = Actor(n_outputs)

        # Critic
        self.critic = MultiCritic(
            n_quantiles=n_quantiles,
            top_quantiles_to_drop=top_quantiles_to_drop,
            n_critics=n_critics,
        )

    def train_step(self, data):
        # Re-new noise matrix every update of 'log_std' params
        self.actor.reset_noise()

        # Set 'Alpha'
        self.alpha.assign(tf.exp(self.log_alpha))

        # Update 'Critic'
        with tf.GradientTape() as tape:
            # concatenating the two batches
            merged_observation = tf.concat(
                [data["observation"], data["next_observation"]], axis=0
            )
            next_action, next_log_pi = self.actor(
                data["next_observation"],
                with_log_prob=True,
                deterministic=False,
            )
            merged_action = tf.concat([data["action"], next_action], axis=0)

            # get Q-value
            quantiles, next_quantiles = tf.split(
                self.critic([merged_observation, merged_action], training=True),
                num_or_size_splits=2,
                axis=0,
            )
            next_quantiles = tf.sort(
                tf.reshape(next_quantiles, [next_quantiles.shape[0], -1])
            )
            next_quantiles = next_quantiles[
                :, : self.critic.quantiles_total - self.critic.top_quantiles_to_drop
            ]

            # Bellman Equation
            target_quantiles = tf.stop_gradient(
                data["reward"]
                + (1.0 - data["terminal"])
                * self.gamma
                * (next_quantiles - self.alpha * next_log_pi)
            )

            # Compute critic loss
            pairwise_delta = (
                target_quantiles[:, tf.newaxis, tf.newaxis, :]
                - quantiles[:, :, :, tf.newaxis]
            )  # batch_size, n_critics, n_quantiles, n_target_quantiles
            abs_pairwise_delta = tf.math.abs(pairwise_delta)
            huber_loss = tf.where(
                abs_pairwise_delta > 1.0,
                abs_pairwise_delta - 0.5,
                pairwise_delta ** 2 * 0.5,
            )

            critic_loss = tf.nn.compute_average_loss(
                tf.reduce_mean(
                    tf.math.abs(
                        self.tau[tf.newaxis, tf.newaxis, :, tf.newaxis]
                        - tf.cast(pairwise_delta < 0.0, dtype=tf.float32)
                    )
                    * huber_loss,
                    axis=[1, 2, 3],
                )
            )

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.critic.trainable_variables)
        )

        # Update 'Actor' & 'Alpha'
        with tf.GradientTape(persistent=True) as tape:
            quantiles, log_pi = self(data["observation"])

            # Compute alpha loss
            alpha_loss = tf.nn.compute_average_loss(
                -self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )

            # Compute actor loss
            actor_loss = tf.nn.compute_average_loss(
                self.alpha * log_pi
                - tf.reduce_mean(
                    tf.reduce_mean(quantiles, axis=2), axis=1, keepdims=True
                )
            )

        gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables)
        )

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
        }

    def call(self, inputs):
        action, log_pi = self.actor(inputs, with_log_prob=True, deterministic=False)
        quantiles = self.critic([inputs, action], training=False)
        return [quantiles, log_pi]

    def compile(self, actor_optimizer, critic_optimizer, alpha_optimizer):
        super(ActorCritic, self).compile()
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
