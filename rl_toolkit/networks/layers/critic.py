from tensorflow.keras.layers import Layer, Dense, Add, Activation, LayerNormalization

import tensorflow as tf


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        self.input_norm = LayerNormalization()

        # 1. layer
        self.fc1_a = Dense(
            400,
            kernel_initializer="he_uniform",
            name="critic_fc1_a",
        )
        self.fc1_norm = LayerNormalization()
        self.fc1 = Activation("relu")

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
        self.fc3_a = Add()
        self.fc3_norm = LayerNormalization()
        self.fc3 = Activation("relu")

        # Output layer
        self.Q_value = Dense(
            1,
            kernel_initializer="glorot_uniform",
            name="Q_value",
        )

    def call(self, inputs):
        x_s = self.input_norm(inputs[0])

        x_s = self.fc1_a(x_s)
        x_s = self.fc1_norm(x_s)
        x_s = self.fc1(x_s)

        x_s = self.fc2_a(x_s)
        x_a = self.fc2_b(inputs[1])

        x = self.fc3_a([x_s, x_a])
        x = self.fc3_norm(x)
        x = self.fc3(x)

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
