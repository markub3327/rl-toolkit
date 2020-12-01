from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp


# Trieda hraca
class Actor:

    def __init__(self, state_shape, action_shape, lr):
        state_input = Input(shape=state_shape, name='state_input')
        l1 = Dense(400, activation='swish', name='h1')(state_input)
        l2 = Dense(300, activation='swish', name='h2')(l1)
        
        # vystupna vrstva   -- 'mu' musi byt v intervale (-∞, ∞), 'sigma' musi byt v intervale (0, ∞)
        mu_l = Dense(action_shape[0], activation='linear', name='mu')(l2)
        log_std_l = Dense(action_shape[0], activation='softplus', name='sigma')(l2)

        # Vytvor model
        self.model = Model(inputs=state_input, outputs=[mu_l, log_std_l])

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    @tf.function
    def predict(self, x, with_logprob=True):
        mu, sigma = self.model(x)

        # Squashed Normal distribution
        pi_distribution = tfp.distributions.Normal(mu, sigma)
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
        l1 = Dense(400, activation='swish', name='h1')(merged)
        l2 = Dense(300, activation='swish', name='h2')(l1)

        # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)!!!
        output = Dense(1, activation='linear', name='q_val')(l2)

        # Vytvor model
        self.model = Model(inputs=[state_input, action_input], outputs=output)

        # Skompiluj model
        self.optimizer = Adam(learning_rate=lr)

        self.model.summary()

    def save(self):
        plot_model(self.model, to_file='model_C.png')
