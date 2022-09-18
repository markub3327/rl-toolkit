import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Activation, Add, Dense


class Counter(Model):
    """
    Counter
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer

    References:
        - [DORA The Explorer: Directed Outreaching Reinforcement Action-Selection](https://arxiv.org/abs/1804.04012)
    """

    def __init__(self, units: list, **kwargs):
        super(Counter, self).__init__(**kwargs)

        # 1. layer
        self.fc_0 = Dense(
            units=units[0],
            activation="relu",
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )

        # 2. layer     TODO(markub3327): Transformer
        self.fc_1 = Dense(
            units=units[1],
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )
        self.fc_2 = Dense(
            units=units[1],
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )
        self.add_0 = Add()
        self.activ_0 = Activation("relu")

        # Output layer
        self.e_value = Dense(
            1,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="e_value",
        )
        self.activ_1 = Activation("sigmoid")

    def call(self, inputs):
        # 1. layer
        state = self.fc_0(inputs[0])

        # 2. layer
        state = self.fc_1(state)
        action = self.fc_2(inputs[1])
        x = self.add_0([state, action])
        x = self.activ_0(x)

        # Output layer
        e_value = self.e_value(x)

        # Intrinsic Reward
        counter = 1.0 / tf.math.sqrt(-tf.math.log_sigmoid(e_value))

        # E-value
        e_value = self.activ_1(e_value)

        return counter, e_value
