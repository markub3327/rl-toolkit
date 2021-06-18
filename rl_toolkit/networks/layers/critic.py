from tensorflow.keras.layers import Layer, Dense, Add, Activation, BatchNormalization

import tensorflow as tf


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # normalize observations
        self.observation_norm = BatchNormalization(scale=False)

        # 1. layer
        self.fc1 = Dense(
            400,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="critic_fc1",
        )
        self.fc1_activ = Activation("relu")
        self.fc1_norm = BatchNormalization(scale=False)

        # 2. layer
        self.fc2_a = Dense(
            300,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="critic_fc2_a",
        )
        self.fc2_b = Dense(
            300,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="critic_fc2_b",
        )

        # Merge state branch and action branch
        self.fc3 = Add()
        self.fc3_activ = Activation("relu")
        self.fc3_norm = BatchNormalization(scale=False)

        # Output layer
        self.Q_value = Dense(
            1,
            kernel_initializer="glorot_uniform",
            name="Q_value",
        )

    def call(self, inputs, training=None):
        x_s = self.observation_norm(inputs[0], training)

        x_s = self.fc1(x_s)
        x_s = self.fc1_activ(x_s)
        x_s = self.fc1_norm(x_s, training)

        x_s = self.fc2_a(x_s)
        x_a = self.fc2_b(inputs[1])

        x = self.fc3([x_s, x_a])
        x = self.fc3_activ(x)
        x = self.fc3_norm(x, training)

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
