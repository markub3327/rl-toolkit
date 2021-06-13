from tensorflow.keras.layers import Layer, Dense, Concatenate

import tensorflow as tf


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        self.fc1 = Dense(
            400,
            activation="relu",
            kernel_initializer="he_uniform",
            name="critic_fc1",
        )
        self.fc2 = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
            name="critic_fc2",
        )

        self.Q_value = Dense(
            1,
            activation="linear",
            name="Q_value",
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
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

        self.merged = Concatenate()

        # Critic
        self.models = []
        for i in range(num_of_critics):
            self.models.append(Critic())

    def call(self, inputs):
        merged = self.merged(inputs)
        Q_values = tf.stack(list(model(merged) for model in self.models), axis=1)
        return Q_values
