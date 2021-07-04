import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense


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

        # 1. layer
        self.fc1 = Dense(
            512,
            kernel_initializer="he_uniform",
        )
        self.fc1_activ = Activation("relu")
        self.fc1_norm = BatchNormalization(momentum=0.0, scale=False)

        # 2. layer
        self.fc2_a = Dense(
            512,
            kernel_initializer="he_uniform",
        )
        self.fc2_b = Dense(
            512,
            kernel_initializer="he_uniform",
        )

        # Merge state branch and action branch
        self.fc2 = Add()
        self.fc2_activ = Activation("relu")
        self.fc2_norm = BatchNormalization(momentum=0.0, scale=False)

        # Output layer
        self.Z = Dense(
            n_quantiles,
            kernel_initializer="glorot_uniform",
            name="Z",
        )

    def call(self, inputs, training=None):
        # 1. layer
        x_s = self.fc1(inputs[0])
        x_s = self.fc1_activ(x_s)
        x_s = self.fc1_norm(x_s, training=training)

        # 2. layer
        x_s = self.fc2_a(x_s)
        x_a = self.fc2_b(inputs[1])

        x = self.fc2([x_s, x_a])
        x = self.fc2_activ(x)
        x = self.fc2_norm(x, training=training)

        # Output layer
        Z = self.Z(x)
        return Z


class MultiCritic(Model):
    """
    MultiCritic
    ===============

    Attributes:
        n_critics (int): number of critic networks
        n_quantiles (int): number of predicted quantiles
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
        Z = tf.stack(list(model(inputs) for model in self.models), axis=1)
        return Z
