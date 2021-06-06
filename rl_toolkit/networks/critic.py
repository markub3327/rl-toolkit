from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Minimum
from tensorflow.keras.models import load_model


class TwinCritic:
    """
    TwinCritic
    ===============

    Attributes:
        state_shape: the shape of state space
        action_shape: the shape of action space
        learning_rate (float): learning rate for optimizer
        model_path (str): path to the model
    """

    def __init__(
        self,
        state_shape=None,
        action_shape=None,
        learning_rate: float = None,
        last_kernel_initializer="glorot_uniform",
        model_path: str = None,
    ):

        if model_path is None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name="state_input")
            action_input = Input(shape=action_shape, name="action_input")

            merged = Concatenate()([state_input, action_input])

            # Critic 1
            c1_h1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="c1_h1"
            )(merged)
            c1_h2 = Dense(
                300, activation="relu", kernel_initializer="he_uniform", name="c1_h2"
            )(c1_h1)

            # Critic 2
            c2_h1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="c2_h1"
            )(merged)
            c2_h2 = Dense(
                300, activation="relu", kernel_initializer="he_uniform", name="c2_h2"
            )(c2_h1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            q1_value = Dense(
                1,
                activation="linear",
                name="Q1_value",
                kernel_initializer=last_kernel_initializer,
            )(c1_h2)
            q2_value = Dense(
                1,
                activation="linear",
                name="Q2_value",
                kernel_initializer=last_kernel_initializer,
            )(c2_h2)

            # output is minimum of Q-values
            output = Minimum(name="Q_value")([q1_value, q2_value])

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Optimalizator modelu
        if learning_rate is not None:
            self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()
