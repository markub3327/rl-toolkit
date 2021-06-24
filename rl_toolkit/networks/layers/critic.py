import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Dense, Layer, LayerNormalization


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:

    References:
        - [CrossNorm: On Normalization for Off-Policy TD Reinforcement Learning](https://arxiv.org/abs/1902.05605)
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # 1. layer
        self.fc1 = Dense(
            400,
            kernel_initializer="he_uniform",
        )
        self.fc1_activ = Activation("relu")
        self.fc1_norm = LayerNormalization(center=False, scale=False)

        # 2. layer
        self.fc2_a = Dense(
            300,
            kernel_initializer="he_uniform",
        )
        self.fc2_b = Dense(
            300,
            kernel_initializer="he_uniform",
        )

        # Merge state branch and action branch
        self.fc2 = Add()
        self.fc2_activ = Activation("relu")
        self.fc2_norm = LayerNormalization(center=False, scale=False)

        # Output layer
        self.Q_value = Dense(
            1,
            kernel_initializer="glorot_uniform",
            name="Q_value",
        )

    def call(self, inputs, training=None):
        # 1. layer
        x_s = self.fc1(inputs[0])
        x_s = self.fc1_activ(x_s)
        x_s = self.fc1_norm(x_s)

        # 2. layer
        x_s = self.fc2_a(x_s)
        x_a = self.fc2_b(inputs[1])

        x = self.fc2([x_s, x_a])
        x = self.fc2_activ(x)
        x = self.fc2_norm(x)

        # Output layer
        Q_values = self.Q_value(x)
        return Q_values


class MultiCritic(Layer):
    """
    MultiCritic
    ===============

    Attributes:
        num_of_critics (int): number of critic networks
    """

    def __init__(self, num_of_critics: int, **kwargs):
        super(MultiCritic, self).__init__(**kwargs)

        # Critic
        self.models = []
        for i in range(num_of_critics):
            self.models.append(Critic())

    def call(self, inputs):
        Q_values = tf.stack(list(model(inputs) for model in self.models), axis=1)
        return Q_values
