import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense


class Critic(Model):
    """
    Critic
    ===============

    Attributes:
        n_quantiles (int): number of predicted quantiles

    References:
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(self, n_quantiles: int, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # Input layer
        self.merged = Concatenate()

        # 1. layer
        self.fc1 = Dense(
            400,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # 2. layer
        self.fc2 = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # Output layer
        self.quantiles = Dense(
            n_quantiles,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="quantiles",
        )

    def call(self, inputs):
        x = self.merged(inputs)

        # 1. layer
        x = self.fc1(x)

        # 2. layer
        x = self.fc2(x)

        # Output layer
        quantiles = self.quantiles(x)
        return quantiles


class MultiCritic(Model):
    """
    MultiCritic
    ===============

    Attributes:
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
    """

    def __init__(
        self, n_quantiles: int, top_quantiles_to_drop: int, n_critics: int, **kwargs
    ):
        super(MultiCritic, self).__init__(**kwargs)

        self.n_quantiles = n_quantiles
        self.quantiles_total = n_quantiles * n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # init critics
        self.models = []
        for i in range(n_critics):
            self.models.append(Critic(n_quantiles))

    def call(self, inputs):
        quantiles = tf.stack(list(model(inputs) for model in self.models), axis=1)
        return quantiles
