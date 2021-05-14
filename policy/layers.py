from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

import tensorflow as tf
import tensorflow_probability as tfp


class NoisyLayer(Layer):
    """
    Noisy layer (gSDE)
    ===========

    Paper: https://arxiv.org/pdf/2005.05719.pdf

    Attributes:
        units (int): number of noisy neurons
        log_std_init (float): initialization value of standard deviation
    """

    def __init__(self, units, log_std_init: float = -3.0, **kwargs):
        super(NoisyLayer, self).__init__(**kwargs)
        self.units = units
        self.log_std_init = log_std_init

    def build(self, input_shape):
        self.log_std = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers.Constant(value=self.log_std_init),
            trainable=True,
            name="log_std",
        )
        self.exploration_mat = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers.Zeros(),
            trainable=False,
            name="exploration_mat",
        )

        self.sample_weights()

    def call(self, inputs):
        return tf.matmul(inputs, self.exploration_mat)

    def get_config(self):
        config = super(NoisyLayer, self).get_config()
        config.update({"units": self.units})
        config.update({"log_std_init": self.log_std_init})
        return config

    def get_std(self):
        # expln
        return tf.where(
            self.log_std <= 0,
            tf.exp(self.log_std),
            tf.math.log1p(self.log_std + 1e-6) + 1.0,
        )

    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.log_std), scale_diag=self.get_std()
        )
        self.exploration_mat.assign(w_dist.sample())
