from tensorflow.keras.layers import Layer, Dense, Concatenate


class Critic(Layer):
    """
    Critic
    ===============

    Attributes:
    """

    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        self.merged = Concatenate()

        self.fc1 = Dense(
            400, activation="relu", kernel_initializer="he_uniform", name="fc1"
        )
        self.fc2 = Dense(
            300, activation="relu", kernel_initializer="he_uniform", name="fc2"
        )

        self.q_value = Dense(
            1,
            activation="linear",
            name="Q2_value",
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs):
        x = self.merged(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.q_value(x)
        return q_values
