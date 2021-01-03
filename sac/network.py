from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


# Trieda hraca
class Actor:

    def __init__(
        self,
        model_path: str,
        log_std_init: float = -3.0,
    ):

        # Nacitaj model
        self.model = load_model(model_path)
        print('Actor loaded from file succesful ...')            

        action_shape = self.model.layers[-1].output_shape
        print(action_shape)
        print(self.model.layers[-1].name)

        # variance params
        self.log_std = tf.Variable(tf.ones([300, action_shape[-1]]) * log_std_init, trainable=True, name='log_std')
        self.exploration_mat = tf.Variable(tf.zeros_like(self.log_std), trainable=False, name='exploration_mat')
        #print(self.log_std)

        # sample new noise matrix
        self.sample_weights()
        #print(self.exploration_mat)

        self.model.summary()

        # Prenosova funkcia vystupnej distribucie
        self.bijector = tfp.bijectors.Tanh()

    @tf.function
    def sample_weights(self):
        # get scale (0, ∞)
        std = tf.exp(self.log_std)

        # create distribution
        w_dist = tfp.distributions.Normal(tf.zeros_like(std), std)
        self.exploration_mat.assign(w_dist.sample())        # save samples

    @tf.function
    def predict(self, x, deterministic=False):
        mean, latent_sde = self.model(x)

        if deterministic:
            pi_action = mean        #  !! zabudol som na bijector pre testovanie rozsahu mean
        else:
            noise = tf.matmul(latent_sde, self.exploration_mat)
            pi_action = self.bijector.forward(mean + noise)

        return pi_action

    def save(self):
        plot_model(self.model, to_file='img/model_A_SAC.png')