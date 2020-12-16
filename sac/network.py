from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


# Trieda hraca
class Actor:

    def __init__(self, state_shape=None, action_shape=None, lr=None, model_path=None):
        if model_path == None:
            state_input = Input(shape=state_shape, name='state_input')
            l1 = Dense(400, activation='relu', name='h1')(state_input)
            l2 = Dense(300, activation='relu', name='h2')(l1)
        
            # vystupna vrstva   -- 'mean' musi byt v intervale (-∞, ∞), 'std' musi byt v intervale (0, ∞)
            mean = Dense(action_shape[0], activation='linear', name='mean')(l2)
            scale = Dense(action_shape[0], activation=None, name='log_std')(l2)
            scale = Lambda(lambda x: tf.math.softplus(x) + 1e-5, name='std')(scale)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, scale])
        else:
            # Nacitaj model
            self.model = tf.keras.models.load_model(model_path)
            print('Actor loaded from file succesful ...')            

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    @tf.function
    def predict(self, x, with_logprob=True, deterministic=False):
        mean, std = self.model(x)

        # Squashed Normal distribution
        if deterministic:
            # Using at test time.
            pi_action = mean
            logp_pi = None
        else:
            pi_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=mean, 
                scale_diag=std
            )
            pi_distribution = tfp.bijectors.Tanh()(pi_distribution)
            pi_action = pi_distribution.sample()

            if with_logprob:
                logp_pi = pi_distribution.log_prob(pi_action)[..., tf.newaxis]
            else:
                logp_pi = None

        return pi_action, logp_pi

    def save(self):
        plot_model(self.model, to_file='img/model_A.png')


# Trieda kritika
class Critic:

    def __init__(self, state_shape=None, action_shape=None, lr=None, model_path=None):
        if model_path == None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name='state_input')
            action_input = Input(shape=action_shape, name='action_input')

            merged = Concatenate()([state_input, action_input])
            l1 = Dense(400, activation='relu', name='h1')(merged)
            l2 = Dense(300, activation='relu', name='h2')(l1)

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
        plot_model(self.model, to_file='img/model_C.png')