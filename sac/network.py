from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Trieda hraca
class Actor:

    def __init__(self, state_shape, action_shape, lr):
        state_input = Input(shape=state_shape, name='state_input')
        l1 = Dense(400, activation='relu', use_bias=True, kernel_initializer='he_uniform', name='h1')(state_input)
        l2 = Dense(300, activation='relu', use_bias=True, kernel_initializer='he_uniform', name='h2')(l1)
        
        # vystupna vrstva   -- musi byt tanh pre (-1,1) ako posledna vrstva!!!
        mu_l = Dense(action_shape[0], activation='linear', use_bias=True, kernel_initializer='glorot_uniform', name='mu')(l2)
        log_std_l = Dense(action_shape[0], activation='linear', use_bias=True, kernel_initializer='glorot_uniform', name='log_std')(l2)

        # Vytvor model
        self.model = Model(inputs=state_input, outputs=[mu_l, log_std_l])

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    @tf.function
    def predict(self, x, with_logprob=True):
        mu, log_std = self.model(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        # Squashed Normal distribution
        pi_distribution = tfp.distributions.Normal(mu, std)
        pi_distribution = tfp.bijectors.Tanh()(pi_distribution)
        #if deterministic:
            # Only used for evaluating policy at test time.
        #    pi_action = mu
        #else:
        pi_action = pi_distribution.sample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action)
        else:
            logp_pi = None

        #tf.print(pi_distribution)
        #tf.print(pi_action)
        #tf.print(logp_pi)      # entropy

        return pi_action, logp_pi

    def save(self):
        plot_model(self.model, to_file='model_A.png')


# Trieda kritika
class Critic:

    def __init__(self, state_shape, action_shape, lr):
        # vstupna vsrtva
        state_input = Input(shape=state_shape, name='state_input')
        action_input = Input(shape=action_shape, name='action_input')

        merged = Concatenate()([state_input, action_input])
        l1 = Dense(400, activation='relu', use_bias=True, kernel_initializer='he_uniform', name='h1')(merged)
        l2 = Dense(300, activation='relu', use_bias=True, kernel_initializer='he_uniform', name='h2')(l1)

        # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
        output = Dense(1, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', name='q_val')(l2)

        # Vytvor model
        self.model = Model(inputs=[state_input, action_input], outputs=output)

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='model_C.png')
