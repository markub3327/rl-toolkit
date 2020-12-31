from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


# Trieda hraca
class Actor:

    def __init__(
        self,
        state_shape = None,
        action_shape = None,
        learning_rate = None,
        model_path = None,
        log_std_init: float = -3.0,
        clip_mean: float = 3.0
    ):

        if model_path == None:
            state_input = Input(shape=state_shape, name='state_input')
            l1 = Dense(400, activation='relu', name='h1')(state_input)
            l2 = Dense(300, activation='relu', name='h2')(l1)       # latent_sde
        
            # vystupna vrstva   -- 'mean' musi byt v intervale (-∞, ∞)
            mean = Dense(action_shape[0], activation='linear', name='mean')(l2)
            mean = Lambda(lambda x: tf.clip_by_value(x, -clip_mean, clip_mean), name='clip_mean')(mean)

            # variance params
            self.log_std = tf.Variable(tf.ones([300, action_shape[0]]) * log_std_init, trainable=True, name='log_std')
            self.exploration_mat = tf.Variable(tf.zeros_like(self.log_std), trainable=False, name='exploration_mat')
            #print(self.log_std)

            # sample new noise matrix
            self.sample_weights()
            #print(self.exploration_mat)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, l2])
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print('Actor loaded from file succesful ...')            

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)
        self.bijector = tfp.bijectors.Tanh()

        self.model.summary()

    @tf.function
    def sample_weights(self):
        # get scale (0, ∞)
        std = tf.exp(self.log_std)

        w_dist = tfp.distributions.Normal(tf.zeros_like(std), std)
        self.exploration_mat.assign(w_dist.sample())

    @tf.function
    def predict(self, x, with_logprob=True, deterministic=False):
        mean, latent_sde = self.model(x)

        if deterministic:
            pi_action = mean
            logp_pi = None
        else:
            noise = tf.matmul(latent_sde, self.exploration_mat)
            pi_action = self.bijector.forward(mean + noise)

            if with_logprob:
                variance = tf.matmul(tf.square(latent_sde), tf.square(tf.exp(self.log_std)))
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.Normal(mean, tf.sqrt(variance + 1e-6)),
                    bijector=self.bijector
                )
                logp_pi = pi_distribution.log_prob(pi_action)
                # sum independent log_probs
                logp_pi = tf.reduce_sum(logp_pi, axis=1, keepdims=True)
            else:
                logp_pi = None

        return pi_action, logp_pi

    def save(self):
        plot_model(self.model, to_file='img/model_A_SAC.png')


# Trieda kritika
class Critic:

    def __init__(
        self,
        state_shape = None,
        action_shape = None,
        learning_rate = None,
        model_path = None
    ):

        if model_path == None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name='state_input')
            action_input = Input(shape=action_shape, name='action_input')

            merged = Concatenate()([state_input, action_input])
            l1 = Dense(400, activation='relu', name='h1')(merged)
            l2 = Dense(300, activation='relu', name='h2')(l1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            output = Dense(1, activation='linear', name='q_val')(l2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print('Critic loaded from file succesful ...')
        
        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='img/model_C_SAC.png')