from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import tensorflow as tf

# Trieda hraca
class Actor:
    def __init__(
        self, 
        noise_type: str,
        action_noise: float,
        state_shape=None, 
        action_shape=None, 
        lr=None, 
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
            self.model = tf.keras.models.load_model(model_path)
            print("Actor loaded from file succesful ...")

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        # select noise generator
        if noise_type == "normal":
            self.noise = NormalActionNoise(
                mean=0.0, sigma=action_noise, size=self.model.output_shape
            )
        elif noise_type == "ornstein-uhlenbeck":
            self.noise = OrnsteinUhlenbeckActionNoise(
                mean=0.0, sigma=action_noise, size=self.model.output_shape
            )
        else:
            raise NameError(f"'{noise_type}' noise is not defined")
        print(f'self.model.output_shape: {self.model.output_shape}')

        self.model.summary()

    @tf.function
    def reset_noise(self):
        self.noise.reset()

# Trieda kritika
class Critic:
    def __init__(self, state_shape=None, action_shape=None, lr=None, model_path=None):

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
            self.model = tf.keras.models.load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()
