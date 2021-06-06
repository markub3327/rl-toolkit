from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import load_model
from rl_toolkit.networks.layers import NoisyLayer

import tensorflow as tf
import tensorflow_probability as tfp


class Actor:
    """
    Actor
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
        learning_rate: float = None,
        clip_mean: float = 2.0,
        last_kernel_initializer="glorot_uniform",
        model_path: str = None,
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
            mean = Dense(
                action_shape[0],
                activation="linear",
                name="mean",
                kernel_initializer=last_kernel_initializer,
            )(latent_sde)
            mean = Lambda(
                lambda x: tf.clip_by_value(x, -clip_mean, clip_mean), name="clip_mean"
            )(mean)

            self._noisy_l = NoisyLayer(action_shape[0], name="noise")
            noise = self._noisy_l(latent_sde)

            # Vytvor model
            self.model = Model(inputs=state_input, outputs=[mean, noise, latent_sde])
        else:
            # Nacitaj model
            self.model = load_model(
                model_path, custom_objects={"NoisyLayer": NoisyLayer}, compile=False
            )
            print("Actor loaded from file succesful ...")

        # Optimalizator modelu
        if learning_rate is not None:
            self.optimizer = Adam(learning_rate=learning_rate)

        # Vystup musi byt v intervale (-1, 1)
        self._bijector = tfp.bijectors.Tanh()

        self.model.summary()

    @tf.function(jit_compile=True)
    def reset_noise(self):
        self._noisy_l.sample_weights()

    def predict(self, x, with_logprob=True, deterministic=False):
        mean, noise, latent_sde = self.model(x)

        if deterministic:
            pi_action = self._bijector.forward(mean)
            logp_pi = None
        else:
            pi_action = self._bijector.forward(mean + noise)

            if with_logprob:
                variance = tf.matmul(
                    tf.square(latent_sde), tf.square(self._noisy_l.get_std())
                )
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.sqrt(variance + 1e-6)
                    ),
                    bijector=self._bijector,
                )
                logp_pi = pi_distribution.log_prob(pi_action)[..., tf.newaxis]
            else:
                logp_pi = None

        return pi_action, logp_pi

    @tf.function(jit_compile=True)
    def get_action(self, state, deterministic):
        a, _ = self.predict(
            tf.expand_dims(state, axis=0),
            with_logprob=False,
            deterministic=deterministic,
        )
        return tf.squeeze(a, axis=0)  # remove batch_size dim
