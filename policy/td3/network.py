from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import load_model


class Actor:
    """
    Actor (for TD3)
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
        learning_rate=None,
        model_path=None,
    ):

        if model_path == None:
            state_input = Input(shape=state_shape, name="state_input")
            l1 = Dense(400, activation="relu", name="h1")(state_input)
            l2 = Dense(300, activation="relu", name="h2")(l1)

            # vystupna vrstva   -- musi byt tanh pre (-1,1) ako posledna vrstva!!!
            output = Dense(action_shape[0], activation="tanh", name="action")(l2)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path, compile=False)
            print("Actor loaded from file succesful ...")

        # Skompiluj model
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()


class Critic:
    """
    Critic (for TD3)
    ===============

    Attributes:
        state_shape: the shape of state space
        action_shape: the shape of action space
        learning_rate (float): learning rate for optimizer
        model_path (str): path to the model
    """

    def __init__(
        self, state_shape=None, action_shape=None, learning_rate=None, model_path=None
    ):

        if model_path == None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name="state_input")
            action_input = Input(shape=action_shape, name="action_input")

            merged = Concatenate()([state_input, action_input])
            l1 = Dense(400, activation="relu", name="h1")(merged)
            l2 = Dense(300, activation="relu", name="h2")(l1)

            # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
            output = Dense(1, activation="linear", name="q_val")(l2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Skompiluj model
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()
