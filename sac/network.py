from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


# Trieda hraca
class Actor:

    def __init__(self, state_shape=None, action_shape=None, lr=None, model_path=None):
        if model_path == None:
            state_input = Input(shape=state_shape, name='state_input')
            l1 = Dense(400, activation='swish', name='h1')(state_input)
            l2 = Dense(300, activation='swish', name='h2')(l1)
        
            # vystupna vrstva   -- 'mu' musi byt v intervale (-∞, ∞), 'sigma' musi byt v intervale (0, ∞)
            mu_l = Dense(action_shape[0], activation='linear', name='mu')(l2)
            std_l = Dense(action_shape[0], activation='softplus', name='sigma')(l2)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mu_l, std_l])
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print('Actor loaded from file succesful ...')

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    @tf.function
    def predict(self, x, with_logprob=True):
        mu, sigma = self.model(x)

        # Squashed Normal distribution
        pi_distribution = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        pi_distribution = tfp.bijectors.Tanh()(pi_distribution)
        
        pi_action = pi_distribution.sample()
        #tf.print(f'action: {pi_action}, {pi_action.shape}')

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action)
            logp_pi = tf.expand_dims(logp_pi, axis=1)       # convert to [batch_size, 1]
        #   tf.print(f'logp_pi: {logp_pi}, {logp_pi.shape}')
        else:
            logp_pi = None

        return pi_action, logp_pi

    def save(self):
        plot_model(self.model, to_file='model_A.png')


# Trieda kritika
class Critic:

    def __init__(self, state_shape=None, action_shape=None, lr=None, model_path=None):
        if model_path == None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name='state_input')
            action_input = Input(shape=action_shape, name='action_input')

            merged = Concatenate()([state_input, action_input])
            l1 = Dense(400, activation='swish', name='h1')(merged)
            l2 = Dense(300, activation='swish', name='h2')(l1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)!!!
            output = Dense(1, activation='linear', name='q_val')(l2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print('Critic loaded from file succesful ...')

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='model_C.png')
