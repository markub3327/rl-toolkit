import os

import numpy as np
import reverb
import wandb
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

from rl_toolkit.networks.callbacks import AgentCallback
from rl_toolkit.networks.models import GAN as GANModel
from rl_toolkit.utils import make_reverb_dataset

from .process import Process


class GAN(Process):
    """
    GAN
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
        train_steps (int): number of training steps
        batch_size (int): size of mini-batch used for training
        actor_units (list): list of the numbers of units in each Actor's layer
        critic_units (list): list of the numbers of units in each Critic's layer
        actor_learning_rate (float): the learning rate for the Actor's optimizer
        critic_learning_rate (float): the learning rate for the Critic's optimizer
        alpha_learning_rate (float): the learning rate for the Alpha's optimizer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        init_noise (float): initialization of the Actor's noise
        model_path (str): path to the model
        save_path (str): path to the models for saving
        log_interval (int): the logging interval to the console
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        train_steps: int,
        batch_size: int,
        # ---
        gan_units: list,
        latent_dim: int,
        generator_learning_rate: float,
        discriminator_learning_rate: float,
        # ---
        # ---
        model_path: str,
        save_path: str,
        # ---
        log_interval: int,
    ):
        super(GAN, self).__init__(env_name, False)

        self._train_steps = train_steps
        self._save_path = save_path
        self._log_interval = log_interval
        self._db_server = db_server

        # Init GAN network
        self.model = GANModel(
            units=gan_units,
            latent_dim=latent_dim,
            n_inputs=np.prod(self._env.observation_space.shape),
        )
        self.model.compile(
            d_optimizer=Adam(
                learning_rate=discriminator_learning_rate,
                beta_1=0.5,
                beta_2=0.999,
            ),
            g_optimizer=Adam(
                learning_rate=generator_learning_rate,
                beta_1=0.5,
                beta_2=0.999,
            ),
        )

        if model_path is not None:
            self.model.load_weights(model_path)

        # Show models details
        self.model.summary()

        # Initializes the reverb's dataset
        self.dataset = make_reverb_dataset(
            server_address=self._db_server,
            table="counter",
            batch_size=batch_size,
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.train_steps = train_steps
        wandb.config.batch_size = batch_size
        wandb.config.gan_units = gan_units
        wandb.config.latent_dim = latent_dim
        wandb.config.generator_learning_rate = generator_learning_rate
        wandb.config.discriminator_learning_rate = discriminator_learning_rate

    def run(self):
        self.model.fit(
            self.dataset,
            epochs=self._train_steps,
            steps_per_epoch=1,
            verbose=0,
            callbacks=[WandbCallback(save_model=False)],
        )

    def save(self):
        if self._save_path:
            # create path if not exists
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)

            # Save model
            self.model.save_weights(os.path.join(self._save_path, "gan.h5"))

    def close(self):
        super(GAN, self).close()

        # create the checkpoint of the database
        client = reverb.Client(self._db_server)
        client.checkpoint()
