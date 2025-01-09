import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Dense


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

    def __init__(self, units: list, n_quantiles: int, merge_index: int, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # list of hidden layers
        self.fc_layers = []

        # prepare 'merge_index'
        if merge_index is None:
            raise ValueError("merge_index must be specified")
        self.merge_index = merge_index

        for i, m in enumerate(units):
            if i != self.merge_index:
                self.fc_layers.append(
                    Dense(units=m, activation="elu")
                )
            else:
                self.fc_layers.append(None)  # add empty layer instead of merge layer

        # 2. layer
        self.fc_state = Dense(
            units=units[self.merge_index],
            activation=None,
        )
        self.fc_action = Dense(
            units=units[self.merge_index],
            activation=None,
        )
        self.add_0 = Add()
        self.activ_0 = Activation("elu")

        # Output layer
        self.quantiles = Dense(
            n_quantiles,
            activation=None,
            name="quantiles",
        )

    def call(self, inputs, training=None):
        # the first hidden layer
        state = inputs[0]
        for layer in self.fc_layers[: self.merge_index]:
            state = layer(state, training=training)

        # the second layer
        state = self.fc_state(state, training=training)
        action = self.fc_action(inputs[1], training=training)  # projection layer
        x = self.add_0([state, action])
        x = self.activ_0(x)

        # the third layer
        for layer in self.fc_layers[(self.merge_index + 1) :]:
            x = layer(x, training=training)

        # Output layer
        return self.quantiles(x, training=training)


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
        merge_index: int = -1,
        **kwargs
    ):
        super(MultiCritic, self).__init__(**kwargs)

        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # init critics
        self.models = [
            Critic(units, n_quantiles, merge_index) for _ in range(n_critics)
        ]

    def call(self, inputs, training=None):
        quantiles = tf.stack(
            [model(inputs, training=training) for model in self.models], axis=1
        )
        return quantiles

    def summary(self):
        for model in self.models:
            model.summary()
        super(MultiCritic, self).summary()
