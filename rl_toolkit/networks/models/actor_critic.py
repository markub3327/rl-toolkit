import tensorflow as tf
from tensorflow.keras import Model

from .actor import Actor
from .critic import MultiCritic
from .gan import GAN


class ActorCritic(Model):
    """
    Actor-Critic
    ===========

    Attributes:
        actor_units (list): list of the numbers of units in each Actor's layer
        critic_units (list): list of the numbers of units in each Critic's layer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        n_outputs (int): number of outputs
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of log_alpha param
        init_noise (float): initialization of Actor's noise

    References:
        - [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(
        self,
        actor_units: list,
        critic_units: list,
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        n_inputs: int,
        n_outputs: int,
        clip_mean_min: float,
        clip_mean_max: float,
        gamma: float,
        tau: float,
        init_alpha: float,
        init_noise: float,
        **kwargs,
    ):
        super(ActorCritic, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(tau)
        self.cum_prob = ((tf.range(n_quantiles, dtype=tf.float32) + 0.5) / n_quantiles)[
            tf.newaxis, tf.newaxis, :, tf.newaxis
        ]

        # init Lagrangian constraint
        self.log_alpha = tf.Variable(init_alpha, trainable=True, name="log_alpha")
        self.target_entropy = tf.cast(-n_outputs, dtype=tf.float32)

        # Actor
        self.actor = Actor(
            units=actor_units,
            n_outputs=n_outputs,
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            init_noise=init_noise,
        )

        # Critic
        self.critic = MultiCritic(
            units=critic_units,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop=top_quantiles_to_drop,
            n_critics=n_critics,
        )

        # Critic target
        self.critic_target = MultiCritic(
            units=critic_units,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop=top_quantiles_to_drop,
            n_critics=n_critics,
        )

        # GAN
        self.gan = GAN(
            units=critic_units,
            latent_dim=64,
            n_inputs=n_inputs,
        )

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.variables, net_targ.variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def train_step(self, sample):
        # Re-new noise matrix every update
        self.actor.reset_noise()

        # Get 'Alpha'
        alpha = tf.math.softplus(self.log_alpha)

        # Get trainable variables
        actor_variables = self.actor.trainable_variables
        critic_variables = self.critic.trainable_variables
        alpha_variables = [self.log_alpha]

        # -------------------- Update 'GAN' -------------------- #
        batch_size = tf.shape(sample.data["observation"])[0]

        # -------------------- Update 'Discriminator' -------------------- #
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            fake_output, _ = self.gan(random_latent_vectors, training=True)
            real_output = self.gan.discriminator(
                sample.data["observation"], training=True
            )

            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.ones_like(fake_output) * (-1), fake_output)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.gan.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.gan.discriminator.trainable_weights)
        )

        # -------------------- Update 'Generator' -------------------- #
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            fake_output, _ = self.gan(random_latent_vectors, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(g_loss, self.gan.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.gan.generator.trainable_weights)
        )

        # -------------------- Update 'Critic' -------------------- #
        next_action, next_log_pi = self.actor(
            sample.data["next_observation"],
            with_log_prob=True,
            deterministic=False,
        )
        next_quantiles = self.critic_target(
            [sample.data["next_observation"], next_action]
        )
        next_quantiles = tf.sort(
            tf.reshape(next_quantiles, [next_quantiles.shape[0], -1])
        )
        next_quantiles = next_quantiles[
            :,
            : self.critic_target.quantiles_total
            - self.critic_target.top_quantiles_to_drop,
        ]

        # Intrinsic Reward
        int_reward = self.gan.discriminator(sample.data["next_observation"])

        # Bellman Equation
        target_quantiles = tf.stop_gradient(
            tf.tanh(sample.data["reward"])
            + int_reward
            + (1.0 - tf.cast(sample.data["terminal"], dtype=tf.float32))
            * self.gamma
            * (next_quantiles - alpha * next_log_pi)
        )

        with tf.GradientTape() as tape:
            quantiles = self.critic([sample.data["observation"], sample.data["action"]])

            # Compute critic loss
            pairwise_delta = (
                target_quantiles[:, tf.newaxis, tf.newaxis, :]
                - quantiles[:, :, :, tf.newaxis]
            )  # batch_size, n_critics, n_quantiles, n_target_quantiles
            critic_loss = tf.nn.compute_average_loss(
                tf.reduce_mean(
                    tf.math.abs(
                        self.cum_prob - tf.cast(pairwise_delta < 0.0, dtype=tf.float32)
                    )
                    * (
                        pairwise_delta
                        + tf.math.softplus(-2.0 * pairwise_delta)
                        - tf.math.log(2.0)
                    ),
                    axis=[1, 2, 3],
                )
            )

        # Compute gradients
        critic_gradients = tape.gradient(critic_loss, critic_variables)

        # Apply gradients
        self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

        # -------------------- Update 'Actor' & 'Alpha' -------------------- #
        with tf.GradientTape(persistent=True) as tape:
            quantiles, log_pi = self(sample.data["observation"])

            # Compute actor loss
            actor_loss = tf.nn.compute_average_loss(
                alpha * log_pi
                - tf.reduce_mean(
                    tf.reduce_mean(quantiles, axis=2), axis=1, keepdims=True
                )
            )

            # Compute alpha loss
            alpha_loss = tf.nn.compute_average_loss(
                -self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )

        # Compute gradients
        actor_gradients = tape.gradient(actor_loss, actor_variables)
        alpha_gradients = tape.gradient(alpha_loss, alpha_variables)

        # Delete the persistent tape manually
        del tape

        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_variables))
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, alpha_variables))

        # -------------------- Soft update target networks -------------------- #
        self._update_target(self.critic, self.critic_target, tau=self.tau)

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "quantiles": quantiles[0],  # logging only one randomly sampled transition
            "log_alpha": self.log_alpha,
            "int_reward": int_reward[0],
            "d_loss": d_loss,
            "g_loss": g_loss,
            "real_output": tf.reduce_mean(real_output),
            "fake_output": tf.reduce_mean(fake_output),
        }

    def call(self, inputs, with_log_prob=True, deterministic=None):
        action, log_pi = self.actor(
            inputs, with_log_prob=with_log_prob, deterministic=deterministic
        )
        quantiles = self.critic([inputs, action])
        return [quantiles, log_pi]

    def compile(
        self,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
        d_optimizer,
        g_optimizer,
    ):
        super(ActorCritic, self).compile()
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def build(self, input_shape):
        super(ActorCritic, self).build(input_shape)

        self._update_target(self.critic, self.critic_target, tau=1.0)

    def summary(self):
        self.actor.summary()
        self.critic.summary()
        self.gan.summary()
