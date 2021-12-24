import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Layer


class MultivariateGaussianNoise(Layer):
    """
    Multivariate Gaussian Noise for exploration
    ===========

    Attributes:
        units (int): number of noisy units
        kernel_initializer: initializer function applied to the `kernel` weights matrix
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        kernel_constraint: constraint function applied to the `kernel` weights matrix

    References:
        - [Generalized State-Dependent Exploration for Deep Reinforcement Learning in Robotics](https://arxiv.org/abs/2005.05719)
    """

    def __init__(
        self,
        units: int,
        kernel_initializer,
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super(MultivariateGaussianNoise, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        super(MultivariateGaussianNoise, self).build(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
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
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
            }
        )

        return config

    @property
    def scale(self):
        return tf.math.softplus(self.kernel)

    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag=(self.scale + 1e-6)
        )
        self.epsilon.assign(w_dist.sample())
