from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers

import tensorflow as tf
import tensorflow_probability as tfp


class NoisyLayer(Layer):
    def __init__(self, units, log_std_init: float = -3.0, **kwargs):
        super(NoisyLayer, self).__init__(**kwargs)
        self.units = units
        self.log_std_init = log_std_init
        self.log_std_initializer = initializers.Constant(value=self.log_std_init)

        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        # Make sure dtype is correct
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )
        input_shape = tf.TensorShape(input_shape)
        self.last_dim = tf.compat.dimension_value(input_shape[-1])

        self.log_std = self.add_weight(
            "log_std",
            shape=[self.last_dim, self.units],
            initializer=self.log_std_initializer,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )

        # sample new noise matrix
        self.sample_weights()
        self.built = True

    def call(self, inputs):
        return tf.matmul(inputs, self.exploration_mat)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def sample_weights(self):
        # get scale (0, ∞)
        std = tf.exp(self.log_std)
        tf.print(self.log_std)
        w_dist = tfp.distributions.Normal(tf.zeros_like(std), std)
        self.exploration_mat = w_dist.sample()

    # Implement get_config to enable serialization. This is optional.
    def get_config(self):
        config = super(NoisyLayer, self).get_config()
        config.update(
            {
                "units": self.units,
                "log_std_init": self.log_std_init,
                "log_std_initializer": initializers.serialize(self.log_std_initializer),
            }
        )
        return config
