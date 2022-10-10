from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, ReLU

import tensorflow as tf
import tensorflow_addons as tfa


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

    def __init__(self, state_shape, latent_dim, units, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Create the discriminator
        self.discriminator = Sequential(
            [
                Input(shape=(state_shape,)),
                tfa.layers.SpectralNormalization(Dense(units[0])),
                ReLU(),
                tfa.layers.SpectralNormalization(Dense(units[1])),
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
                Dense(state_shape, activation="tanh"),
            ],
            name="generator",
        )

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def call(self, inputs, training=None):
        generated_images = self.generator(inputs, training=training)
        score = self.discriminator(generated_images, training=training)
        return score, generated_images

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]

        # -------------------- Update 'Discriminator' -------------------- #
        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            fake_output, _ = self(random_latent_vectors, training=True)
            real_output = self.discriminator(real_images, training=True)

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
        }
