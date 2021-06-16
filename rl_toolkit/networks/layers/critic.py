from tensorflow.keras.layers import Layer, Dense, Add, Activation

import tensorflow as tf


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # 1. layer
        self.fc1_a = Dense(
            400,
            kernel_initializer="he_uniform",
            name="critic_fc1_a",
        )
        self.fc1 = Activation('relu')

        # 2. layer
        self.fc2_a = Dense(
            300,
            kernel_initializer="he_uniform",
            name="critic_fc2_a",
        )
        self.fc2_b = Dense(
            300,
            kernel_initializer="he_uniform",
            name="critic_fc2_b",
        )

        # Merge state branch and action branch
        self.fc2_c = Add()
        self.fc2 = Activation('relu')

        # Output layer
        self.Q_value = Dense(
            1,
            name="Q_value",
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs):
        x_s = self.fc1_a(inputs[0])
        x_s = self.fc1(x_s)

        x_s = self.fc2_a(x_s)
        x_a = self.fc2_b(inputs[1])
        x = self.fc2_c([x_s, x_a])
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

        # Critic
        self.models = []
        for i in range(num_of_critics):
            self.models.append(Critic())

    def call(self, inputs):
        Q_values = tf.stack(list(model(inputs) for model in self.models), axis=1)
        return Q_values
