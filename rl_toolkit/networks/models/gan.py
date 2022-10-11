from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, ReLU
from rl_toolkit.utils import SpectralNormalization


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

    def call(self, inputs, training=None):
        generated_states = self.generator(inputs, training=training)
        score = self.discriminator(generated_states, training=training)
        return score, generated_states

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
