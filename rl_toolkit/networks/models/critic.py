import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense


class Critic(Model):
    """
    Critic
    ===============

    Attributes:
        n_quantiles (int): number of predicted quantiles

    References:
        - [CrossNorm: On Normalization for Off-Policy TD Reinforcement Learning](https://arxiv.org/abs/1902.05605)
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(self, n_quantiles: int, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # Input layer
        self.merged = Concatenate()
        self.merged_norm = BatchNormalization(momentum=0.0, scale=False)

        # 1. layer
        self.fc1 = Dense(
            512,
            activation="relu",
            kernel_initializer="he_uniform",
        )
        self.fc1_norm = BatchNormalization(momentum=0.0, scale=False)

        # 2. layer
        self.fc2 = Dense(
            512,
            activation="relu",
            kernel_initializer="he_uniform",
        )
        self.fc2_norm = BatchNormalization(momentum=0.0, scale=False)

        # 3. layer
        self.fc3 = Dense(
            512,
            activation="relu",
            kernel_initializer="he_uniform",
        )
        self.fc3_norm = BatchNormalization(momentum=0.0, scale=False)

        # Output layer
        self.quantiles = Dense(
            n_quantiles,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="quantiles",
        )

    def call(self, inputs, training=None):
        x = self.merged(inputs)
        x = self.merged_norm(x, training=training)

        # 1. layer
        x = self.fc1(x)
        x = self.fc1_norm(x, training=training)

        # 2. layer
        x = self.fc2(x)
        x = self.fc2_norm(x, training=training)

        # 3. layer
        x = self.fc3(x)
        x = self.fc3_norm(x, training=training)

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
