from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from .noisy_layer import NoisyLayer
from os import path

import time
import os
import tensorflow as tf
import tensorflow_probability as tfp

# Trieda hraca
class Actor:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # Nacitaj model
        self.reload()

        print(self.model.trainable_variables)

        self.model.summary()

        # get log_std layer
        self.noisy_l = self.model.get_layer(name="log_std")

        # Prenosova funkcia vystupnej distribucie
        self.bijector = tfp.bijectors.Tanh()

    @tf.function
    def predict(self, x, deterministic=False):
        mean, noise, latent_sde = self.model(x)
        print(mean)

        if deterministic:
            pi_action = mean  #  !! zabudol som na bijector pre testovanie rozsahu mean
        else:
            pi_action = self.bijector.forward(mean + noise)

        return pi_action

    def save(self):
        plot_model(self.model, to_file="img/model_A_SAC.png")

    def sample_weights(self):
        self.noisy_l.sample_weights()

    def reload(self):
        while True:
            if path.exists(f"{self.model_path}.lock") == False:
                # create lock file
                #f = open(f"{self.model_path}.lock", "w").close()
                # Nacitaj model
                self.model = load_model(self.model_path, custom_objects={"NoisyLayer": NoisyLayer})
                #os.remove(f"{self.model_path}.lock")
                print("Actor loaded from file succesful ... 😊")
                break
            else:
                print("Warning: File is already open")
                time.sleep(0.5)   # 5

