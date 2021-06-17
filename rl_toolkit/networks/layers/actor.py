from rl_toolkit.networks.activations import clipped_linear
from .noise import MultivariateGaussianNoise
from tensorflow.keras.layers import Layer, Dense, Activation, LayerNormalization

import tensorflow as tf
import tensorflow_probability as tfp


class Actor(Layer):
    """
    Actor
    ===============

    Attributes:
        num_of_outputs (int): number of outputs
    """

    def __init__(self, num_of_outputs: int, **kwargs):
        super(Actor, self).__init__(**kwargs)

        self.fc1 = Dense(400, kernel_initializer="he_uniform", name="fc1")
        self.fc1_norm = LayerNormalization()
        self.fc1_activ = Activation("relu")

        self.latent_sde = Dense(
            300,
            kernel_initializer="he_uniform",
            name="latent_sde",
        )
        self.latent_sde_norm = LayerNormalization()
        self.latent_sde_activ = Activation("relu")

        # Deterministicke akcie
        self.mean = Dense(
            num_of_outputs,
            activation=clipped_linear,
            name="mean",
            kernel_initializer="glorot_uniform",
        )

        # Stochasticke akcie
        self.noise = MultivariateGaussianNoise(num_of_outputs, name="noise")

        # Vystupna prenosova funkcia
        self.bijector = tfp.bijectors.Tanh()

    def reset_noise(self):
        self.noise.sample_weights()

    def call(self, inputs, training=None, with_log_prob=None, deterministic=None):
        x = self.fc1(inputs)
        x = self.fc1_norm(x)
        x = self.fc1_activ(x)

        latent_sde = self.latent_sde(x)
        latent_sde = self.latent_sde_norm(latent_sde)
        latent_sde = self.latent_sde_activ(latent_sde)

        mean = self.mean(latent_sde)
        noise = self.noise(latent_sde)

        if deterministic:
            action = self.bijector.forward(mean)
            log_prob = None
        else:
            action = self.bijector.forward(mean + noise)

            if with_log_prob:
                variance = tf.matmul(
                    tf.square(latent_sde), tf.square(self.noise.get_std())
                )
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.sqrt(variance + 1e-6)
                    ),
                    bijector=self.bijector,
                )
                log_prob = pi_distribution.log_prob(action)[..., tf.newaxis]
            else:
                log_prob = None

        return [action, log_prob]
