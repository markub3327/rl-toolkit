import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant, VarianceScaling
from tensorflow.keras.layers import Dense, Lambda

from rl_toolkit.networks.layers import MultivariateGaussianNoise

uniform_initializer = VarianceScaling(distribution="uniform", mode="fan_in", scale=1.0)


class Actor(Model):
    """
    Actor
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        n_outputs (int): number of outputs
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        init_noise (float): initialization of the Actor's noise

    References:
        - [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
    """

    def __init__(
        self,
        units: list,
        n_outputs: int,
        clip_mean_min: float,
        clip_mean_max: float,
        init_noise: float,
        **kwargs
    ):
        super(Actor, self).__init__(**kwargs)

        # 1. layer
        self.fc_0 = Dense(
            units=units[0],
            activation="relu",
            kernel_initializer=uniform_initializer,
        )

        # 2. layer     TODO(markub3327): Transformer
        self.fc_1 = Dense(
            units=units[1],
            activation="relu",
            kernel_initializer=uniform_initializer,
        )

        # Deterministicke akcie
        self.mean = Dense(
            n_outputs,
            activation="linear",
            kernel_initializer=uniform_initializer,
            name="mean",
        )
        self.clip_mean = Lambda(
            lambda x: tf.clip_by_value(x, clip_mean_min, clip_mean_max),
            name="clip_mean",
        )

        # Stochasticke akcie
        self.noise = MultivariateGaussianNoise(
            n_outputs,
            kernel_initializer=Constant(value=init_noise),
            name="noise",
        )

        # Vystupna prenosova funkcia
        self.bijector = tfp.bijectors.Tanh()

    def reset_noise(self):
        self.noise.sample_weights()

    def call(self, inputs, with_log_prob=True, deterministic=None):
        # 1. layer
        x = self.fc_0(inputs)

        # 2. layer
        latent_sde = self.fc_1(x)

        # Output layer
        mean = self.mean(latent_sde)
        mean = self.clip_mean(mean)

        if deterministic:
            action = self.bijector.forward(mean)
            log_prob = None
        else:
            noise = self.noise(latent_sde)
            action = self.bijector.forward(mean + noise)

            if with_log_prob:
                variance = tf.matmul(tf.square(latent_sde), tf.square(self.noise.scale))
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

    # TODO(markub3327):    def train_step(self, data):
