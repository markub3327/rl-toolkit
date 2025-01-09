import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, backend
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Lambda

from rl_toolkit.networks.layers import MultivariateGaussianNoise


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

        self.units = units
        self.n_outputs = n_outputs
        self.clip_mean_min = clip_mean_min
        self.clip_mean_max = clip_mean_max
        self.init_noise = init_noise

        # list of hidden layers
        self.fc_layers = []

        for m in units:
            self.fc_layers.append(
                Dense(units=m, activation="elu")
            )

        # Deterministicke akcie
        self.mean = Dense(
            n_outputs,
            activation=None,
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

    def call(self, inputs, training=None, with_log_prob=True, deterministic=None):
        x = inputs

        # hidden layers
        for layer in self.fc_layers:
            x = layer(x, training=training)

        # output layer
        mean = self.mean(x, training=training)
        mean = self.clip_mean(mean, training=training)

        if deterministic:
            action = self.bijector.forward(mean)
        else:
            noise = self.noise(x, training=training)
            action = self.bijector.forward(mean + noise)

            if with_log_prob:
                variance = tf.matmul(tf.square(x), tf.square(self.noise.scale))
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.sqrt(variance + backend.epsilon())
                    ),
                    bijector=self.bijector,
                )
                log_prob = pi_distribution.log_prob(action)[..., tf.newaxis]

                return [action, log_prob]

        return action

    def get_config(self):
        config = super(Actor, self).get_config()
        config.update(
            {
                "units": self.units,
                "n_outputs": self.n_outputs,
                "clip_mean_min": self.clip_mean_min,
                "clip_mean_max": self.clip_mean_max,
                "init_noise": self.init_noise,
            }
        )

        return config
