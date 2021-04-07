from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import load_model
#from .noisy_layer import NoisyLayer

import tensorflow as tf
import tensorflow_probability as tfp


class Actor:
    """
    Actor (for SAC)
    ===============
    Attributes:
        state_shape: the shape of state space
        action_shape: the shape of action space
        learning_rate (float): learning rate for optimizer
        model_path (str): path to the model
    """

    def __init__(
        self,
        state_shape=None,
        action_shape=None,
        model_path=None,
        learning_rate: float = 3e-4,
#       clip_mean: float = 2.0,
    ):

        if model_path == None:
            state_input = Input(shape=state_shape, name="state_input")

            l1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="h1"
            )(state_input)
            latent_sde = Dense(
                300,
                activation="relu",
                kernel_initializer="he_uniform",
                name="latent_sde",
            )(l1)

            # vystupna vrstva   -- 'mean' musi byt v intervale (-∞, ∞)
            mean = Dense(action_shape[0], activation="linear", name="mean")(latent_sde)
#           mean = Lambda(
#               lambda x: tf.clip_by_value(x, -clip_mean, clip_mean), name="clip_mean"
#           )(mean)

#           self.noisy_l = NoisyLayer(action_shape[0], name="noise")
#           noise = self.noisy_l(latent_sde)

            cov_matrix = Dense((action_shape[0] * (action_shape[0] + 1) // 2), activation="linear", name="cov_matrix")(latent_sde)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, cov_matrix])
        else:
            # Nacitaj model
            self.model = load_model(model_path, compile=False)
            print("Actor loaded from file succesful ...")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)
        self.bijector = tfp.bijectors.Tanh()
        self.scale_tril = tfp.bijectors.FillScaleTriL()

        self.model.summary()

    @tf.function
    def reset_noise(self):
        pass
#       self.noisy_l.sample_weights()

    def predict(self, x, with_logprob=True, deterministic=False):
        mean, cov_m = self.model(x)
        scale_tril = self.scale_tril(cov_m)
        print(mean.shape)
        print(cov_m.shape)
        print(scale_tril.shape)

        if deterministic:
            pi_action = self.bijector.forward(mean)
            logp_pi = None
        else:
            pi_distribution = tfp.distributions.TransformedDistribution(
                distribution=tfp.distributions.MultivariateNormalTriL(
                    loc=mean,
                    scale_tril=scale_tril
                ),
                bijector=self.bijector,
            )
            pi_action = pi_distribution.sample()

            if with_logprob:
                logp_pi = pi_distribution.log_prob(pi_action)[..., tf.newaxis]
            else:
                logp_pi = None

        return pi_action, logp_pi


class Critic:
    """
    Critic (for SAC)
    ===============

    Attributes:
        state_shape: the shape of state space
        action_shape: the shape of action space
        learning_rate (float): learning rate for optimizer
        model_path (str): path to the model
    """

    def __init__(
        self,
        state_shape=None,
        action_shape=None,
        model_path=None,
        learning_rate: float = 3e-4,
    ):

        if model_path == None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name="state_input")
            action_input = Input(shape=action_shape, name="action_input")

            merged = Concatenate()([state_input, action_input])
            l1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="h1"
            )(merged)
            l2 = Dense(
                300, activation="relu", kernel_initializer="he_uniform", name="h2"
            )(l1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            output = Dense(1, activation="linear", name="q_val")(l2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()
