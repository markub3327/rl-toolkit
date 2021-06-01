from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers


class Critic:
    """
    Critic
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
        learning_rate: float = 3e-4,
        model_path: str = None,
    ):

        if model_path is None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name="state_input")
            action_input = Input(shape=action_shape, name="action_input")

            merged = Concatenate()([state_input, action_input])
            h1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="h1"
            )(merged)
            h2 = Dense(
                300, activation="relu", kernel_initializer="he_uniform", name="h2"
            )(h1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            output = Dense(
                1,
                activation="linear",
                name="q_val",
                kernel_initializer=initializers.RandomUniform(
                    minval=-0.03, maxval=0.03
                ),
            )(h2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()
