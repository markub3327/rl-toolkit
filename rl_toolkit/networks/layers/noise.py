import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Layer


class MultivariateGaussianNoise(Layer):
    """
    Noisy layer (gSDE)
    ===========

    Paper: https://arxiv.org/pdf/2005.05719.pdf

    Attributes:
        units (int): number of noisy neurons
        kernel_initializer (float): initialization value of the `kernel` weights matrix
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        kernel_constraint: constraint function applied to the `kernel` weights matrix
    """

    def __init__(
        self,
        units,
        kernel_initializer: float = -3.0,
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super(MultivariateGaussianNoise, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        super(MultivariateGaussianNoise, self).build(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer=initializers.Constant(value=self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.epsilon = self.add_weight(
            name="epsilon",
            shape=(input_shape[-1], self.units),
            initializer=initializers.Zeros(),
            trainable=False,
        )

        # Re-new noise matrix
        self.sample_weights()

    def call(self, inputs):
        return tf.matmul(inputs, self.epsilon)

    def get_config(self):
        config = super(MultivariateGaussianNoise, self).get_config()
        config.update(
            {
                "units": self.units,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
            }
        )

        return config

    def get_std(self):
        # expln
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )

    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag=self.get_std()
        )
        self.epsilon.assign(w_dist.sample())
