import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .noisy_layer import NoisyLayer


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
        clip_mean: float = 2.0,
    ):

        if model_path is None:
            state_input = Input(shape=state_shape, name="state_input")

            h1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="h1"
            )(state_input)
            latent_sde = Dense(
                300,
                activation="relu",
                kernel_initializer="he_uniform",
                name="latent_sde",
            )(h1)

            # vystupna vrstva   -- 'mean' musi byt v intervale (-∞, ∞)
            mean = Dense(action_shape[0], activation="linear", name="mean")(latent_sde)
            mean = Lambda(
                lambda x: tf.clip_by_value(x, -clip_mean, clip_mean), name="clip_mean"
            )(mean)

            self.noisy_l = NoisyLayer(action_shape[0], name="noise")
            noise = self.noisy_l(latent_sde)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, noise, latent_sde])
        else:
            # Nacitaj model
            self.model = load_model(
                model_path, custom_objects={"NoisyLayer": NoisyLayer}, compile=False
            )
            print("Actor loaded from file succesful ...")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)
        self.bijector = tfp.bijectors.Tanh()

        self.model.summary()

    @tf.function
    def reset_noise(self):
        self.noisy_l.sample_weights()

    def predict(self, x, with_logprob=True, deterministic=False):
        mean, noise, latent_sde = self.model(x)

        if deterministic:
            pi_action = self.bijector.forward(mean)
            logp_pi = None
        else:
            pi_action = self.bijector.forward(mean + noise)

            if with_logprob:
                variance = tf.matmul(
                    tf.square(latent_sde), tf.square(self.noisy_l.get_std())
                )
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.sqrt(variance + 1e-6)
                    ),
                    bijector=self.bijector,
                )
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

        if model_path is None:
            # vstupna vsrtva
            state_input = Input(shape=state_shape, name="state_input")
            action_input = Input(shape=action_shape, name="action_input")

            merged = Concatenate()([state_input, action_input])
            h1 = Dense(
                400, activation="relu", kernel_initializer="he_uniform", name="h1"
            )(merged)
            h2 = Dense(
                300, activation="relu", kernel_initializer="he_uniform", name="h2"
            )(h1)

            # vystupna vrstva   -- Q hodnoty su v intervale (-∞, ∞)
            output = Dense(1, activation="linear", name="q_val")(h2)

            # Vytvor model
            self.model = Model(inputs=[state_input, action_input], outputs=output)
        else:
            # Nacitaj model
            self.model = load_model(model_path)
            print("Critic loaded from file succesful ...")

        # Optimalizator modelu
        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.summary()
