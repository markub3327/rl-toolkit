from rl_toolkit.networks.activations import clipped_linear
from rl_toolkit.networks.layers import MultivariateGaussianNoise
from tensorflow.keras.layers import Layer, Dense

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

        self.fc1 = Dense(
            400, activation="relu", kernel_initializer="he_uniform", name="fc1"
        )

        self.latent_sde = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
            name="latent_sde",
        )

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

    def call(self, inputs, with_log_prob=True, deterministic=False):
        x = self.fc1(inputs)
        latent_sde = self.latent_sde(x)
        mean = self.mean(x)
        noise = self.noise(x)

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

        return action, log_prob
