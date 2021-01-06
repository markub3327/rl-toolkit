from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import tensorflow as tf

# Trieda hraca
class Actor:

    def __init__(
        self, 
        model_path: str,
        noise_type: str,
        action_noise: float
    ):

        # Nacitaj model
        self.model = load_model(model_path)
        print('Actor loaded from file succesful ... 😊')

        self.model.summary()

        action_shape = self.model.layers[-1].output_shape
        print(action_shape)
        print(self.model.layers[-1].name)

        # select noise generator
        if (noise_type == 'normal'):
            self.noise = NormalActionNoise(mean=0.0, sigma=action_noise, size=action_shape)
        elif (noise_type == 'ornstein-uhlenbeck'):
            self.noise = OrnsteinUhlenbeckActionNoise(mean=0.0, sigma=action_noise, size=action_shape)
        else:
            raise NameError(f"'{noise_type}' noise is not defined")

    @tf.function
    def predict(self, x, deterministic=False):
        pi_action = self.model(x)

        if deterministic == False:
            pi_action = tf.clip_by_value(pi_action + self.noise.sample(), -1.0, 1.0)

        return pi_action

    def save(self):
        plot_model(self.model, to_file='img/model_A_TD3.png')