from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, ReLU
from rl_toolkit.utils import SpectralNormalization

import tensorflow as tf


class GAN(Model):
    """
    Generative Adversarial Network (GAN)
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer

    References:
        - [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
        - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
        - [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
    """

    def __init__(self, units: list, latent_dim: int, n_inputs: int, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        self.discriminator = Sequential(
            [
                Input(shape=(n_inputs,)),
                SpectralNormalization(Dense(units[0])),
                ReLU(),
                SpectralNormalization(Dense(units[1])),
                ReLU(),
                Dense(1),
            ],
            name="discriminator",
        )
        self.generator = Sequential(
            [
                Input(shape=(latent_dim,)),
                Dense(units[1]),
                ReLU(),
                Dense(units[0]),
                ReLU(),
                Dense(n_inputs),
            ],
            name="generator",
        )

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def call(self, inputs, training=None):
        generated_states = self.generator(inputs, training=training)
        score = self.discriminator(generated_states, training=training)
        return score, generated_states

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    def train_step(self, real_states):
        # Sample random points in the latent space
        batch_size = tf.shape(real_states)[0]

        # -------------------- Update 'Discriminator' -------------------- #
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            fake_output, _ = self(random_latent_vectors, training=True)
            real_output = self.discriminator(real_states, training=True)

            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.ones_like(fake_output) * (-1), fake_output)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # -------------------- Update 'Generator' -------------------- #
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            fake_output, _ = self(random_latent_vectors, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "real_output": tf.reduce_mean(real_output),
            "fake_output": tf.reduce_mean(fake_output),
        }
