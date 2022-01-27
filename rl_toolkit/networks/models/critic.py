import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Activation, Add, Dense

uniform_initializer = VarianceScaling(distribution="uniform", mode="fan_in", scale=1.0)


class Critic(Model):
    """
    Critic
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        n_quantiles (int): number of predicted quantiles

    References:
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(self, units: list, n_quantiles: int, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # 1. layer
        self.fc_0 = Dense(
            units=units[0],
            activation="relu",
            kernel_initializer=uniform_initializer,
        )

        # 2. layer     TODO(markub3327): Transformer
        self.fc_1 = Dense(
            units=units[1],
            kernel_initializer=uniform_initializer,
        )
        self.fc_2 = Dense(
            units=units[1],
            kernel_initializer=uniform_initializer,
        )
        self.add_0 = Add()
        self.activ_0 = Activation("relu")

        # Output layer
        self.quantiles = Dense(
            n_quantiles,
            activation="linear",
            kernel_initializer=uniform_initializer,
            name="quantiles",
        )

    def call(self, inputs):
        # 1. layer
        state = self.fc_0(inputs[0])

        # 2. layer
        state = self.fc_1(state)
        action = self.fc_2(inputs[1])
        x = self.add_0([state, action])
        x = self.activ_0(x)

        # Output layer
        quantiles = self.quantiles(x)
        return quantiles


class MultiCritic(Model):
    """
    MultiCritic
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
    """

    def __init__(
        self,
        units: list,
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        **kwargs
    ):
        super(MultiCritic, self).__init__(**kwargs)

        self.n_quantiles = n_quantiles
        self.quantiles_total = n_quantiles * n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # init critics
        self.models = [Critic(units, n_quantiles) for _ in range(n_critics)]

    def call(self, inputs):
        quantiles = tf.stack([model(inputs) for model in self.models], axis=1)
        return quantiles

    def summary(self):
        for model in self.models:
            model.summary()

    # TODO(markub3327):    def train_step(self, data):
