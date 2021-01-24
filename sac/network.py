from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from .noisy_layer import NoisyLayer

import os
import sys
import time
import errno
import tensorflow as tf
import tensorflow_probability as tfp

# Trieda hraca
class Actor:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # Prenosova funkcia vystupnej distribucie
        self.bijector = tfp.bijectors.Tanh()

    #@tf.function    --- so statickym grafom to pada
    def predict(self, x, deterministic=False):
        pi_action, noise, latent_sde = self.model(x)
        #print(pi_action)

        if deterministic == False:
            pi_action += noise

        return self.bijector.forward(pi_action)

    def save_plot(self):
        plot_model(self.model, to_file="img/model_A_SAC.png")

    def sample_weights(self):
        self.model.layers[-1].sample_weights()

    def load(self):
        # cakaj na zamok
        lockfile = f"{self.model_path}.lock"
        while True:
            try:
                fd = os.open(lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)

                # nacitaj model
                self.model = load_model(self.model_path, custom_objects={"NoisyLayer": NoisyLayer}, compile=False)
                #self.model.summary()

                # uvolni zamok
                os.close(fd)
                os.unlink(lockfile)

                print("Loaded succesful ... 😊")
                break       # ukonci proces zapisu
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

                sys.stdout.write('\rWarning: File is already open by another user...')   
                sys.stdout.flush()

                time.sleep(0.05)