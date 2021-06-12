from tensorflow.keras.layers import Layer, Dense, Concatenate


class TwinCritic(Layer):
    """
    TwinCritic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(TwinCritic, self).__init__(**kwargs)

        self.merged = Concatenate()

        # Critic 1
        self.critic_1_fc1 = Dense(
            400, activation="relu", kernel_initializer="he_uniform", name="critic_1_fc1"
        )
        self.critic_1_fc2 = Dense(
            300, activation="relu", kernel_initializer="he_uniform", name="critic_1_fc2"
        )

        # Critic 2
        self.critic_2_fc1 = Dense(
            400, activation="relu", kernel_initializer="he_uniform", name="critic_2_fc1"
        )
        self.critic_2_fc2 = Dense(
            300, activation="relu", kernel_initializer="he_uniform", name="critic_2_fc2"
        )

        self.Q1_value = Dense(
            1,
            activation="linear",
            name="Q_value",
            kernel_initializer="glorot_uniform",
        )

        self.Q2_value = Dense(
            1,
            activation="linear",
            name="Q_value",
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs):
        merged = self.merged(inputs)

        x = self.critic_1_fc1(merged)
        x = self.critic_1_fc2(x)
        Q1_values = self.Q1_value(x)

        x = self.critic_2_fc1(merged)
        x = self.critic_2_fc2(x)
        Q2_values = self.Q2_value(x)

        return Q1_values, Q2_values
