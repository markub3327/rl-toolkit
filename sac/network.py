from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from .noisy_layer import NoisyLayer

import os
import sys
import time
import errno
import tensorflow as tf
import tensorflow_probability as tfp

# Trieda hraca
class Actor:
    def __init__(
        self,
        state_shape=None,
        action_shape=None,
        learning_rate=None,
        model_path=None,
        clip_mean: float = 2.0,
        epsilon: float = 1e-06
    ):
        self.epsilon = epsilon

        if model_path == None:
            state_input = Input(shape=state_shape, name="state_input")
            l1 = Dense(400, activation="relu", name="h1")(state_input)
            l2 = Dense(300, activation="relu", name="latent_sde")(l1)

            # vystupna vrstva   -- 'mean' musi byt v intervale (-∞, ∞)
            mean = Dense(action_shape[0], activation="linear", name="mean")(l2)
            mean = Lambda(
                lambda x: tf.clip_by_value(x, -clip_mean, clip_mean), name="clip_mean"
            )(mean)

            #log_std = Dense(action_shape[0], name="log_std")(l2)
            #log_std = Lambda(
            #    lambda x: tf.math.softplus(x) + 1e-5
            #)(log_std)

            self._noisy_l1 = NoisyLayer(action_shape[0], name="log_std")
            noisy_l1 = self._noisy_l1(l2)
            config = self._noisy_l1.get_config()
            print(config)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, noisy_l1, l2])
        else:
            # Nacitaj model
            self.model = load_model(model_path, custom_objects={"NoisyLayer": NoisyLayer}, compile=False)
            self._noisy_l1 = self.model.get_layer(name="log_std")
            print("Actor loaded from file succesful ... 😊")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        # Prenosova funkcia vystupnej distribucie
        self.bijector = tfp.bijectors.Tanh()

        self.model.summary()

        # init lockfile
        self.is_locked = False

    @tf.function
    def predict(self, x, with_logprob=True, deterministic=False):
        mean, noise, latent_sde = self.model(x)

        if deterministic:
            pi_action = mean
            logp_pi = None
        else:
            pi_action = self.bijector.forward(mean + noise)
            if with_logprob:
                variance = tf.matmul(tf.square(latent_sde), tf.square(self._noisy_l1.get_std()))
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.Normal(
                        mean, tf.sqrt(variance + self.epsilon)
                    ),
                    bijector=self.bijector
                )
                logp_pi = pi_distribution.log_prob(pi_action)
                # sum independent log_probs
                logp_pi = tf.reduce_sum(logp_pi, axis=1, keepdims=True)
            else:
                logp_pi = None

        return pi_action, logp_pi

    def create_lock(self, path):
        # cakaj na zamok
        self.lockfile = f"{path}.lock"
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)
                self.is_locked = True
                print('Lockfile created')
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                sys.stdout.write('\rWarning: File is already open by another user...')   
                sys.stdout.flush()
                # release medium
                time.sleep(0.05)

    def save_weights(self, path):
        # uloz model
        self.model.save_weights(path)
        print("Saved weights successful 😊")
    

    def load_weights(self, path):
        if os.path.exists(path):
            # nacitaj model
            self.model.load_weights(path)
            print("Loaded succesful ... 😊")

    def release_lock(self):
        if self.is_locked:
            # uvolni zamok
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def sample_weights(self):
        self._noisy_l1.sample_weights()

# Trieda kritika
class Critic:
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

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            output = Dense(1, activation="linear", name="q_val")(l2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print("Critic loaded from file succesful ... 😊")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()