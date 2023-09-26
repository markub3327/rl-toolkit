import reverb
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import LearningRateScheduler
from wandb.keras import WandbCallback

from rl_toolkit.networks.callbacks import AgentCallback, PrintLR, cosine_schedule
from rl_toolkit.networks.models import DuelingDQN
from rl_toolkit.utils import make_reverb_dataset

from ...core.process import Process


class Learner(Process):
    """
    Learner
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
        save_path (str): path to the models for saving
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
        num_layers: int,
        embed_dim: int,
        ff_mult: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        learning_rate: float,
        # ---
        global_clipnorm: float,
        weight_decay: float,
        warmup_steps: int,
        # ---
        gamma: float,
        tau: float,
        # ---
        save_path: str,
    ):
        super(Learner, self).__init__(env_name, False)

        tf.config.optimizer.set_jit(True)  # Enable XLA.

        self._train_steps = train_steps
        self._save_path = save_path
        self._db_server = db_server
        self._warmup_steps = warmup_steps
        action_space = self._env.action_space.n

        # Init Dueling DQN network
        target_dqn_model = DuelingDQN(
            action_space,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ff_mult=ff_mult,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            target_dqn_model=None,
            gamma=gamma,
            tau=tau,
        )
        target_dqn_model.build((None,) + self._env.observation_space.shape)

        self.model = DuelingDQN(
            action_space,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ff_mult=ff_mult,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            target_dqn_model=target_dqn_model,
            gamma=gamma,
            tau=tau,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        dqn_optimizer = tf.keras.optimizers.AdamW(
            global_clipnorm=global_clipnorm,
            weight_decay=weight_decay,
        )
        dqn_optimizer.exclude_from_weight_decay(
            var_names=["bias", "layer_normalization", "position"]
        )
        self.model.compile(optimizer=dqn_optimizer)

        # copy original to target model's weights
        target_dqn_model.set_weights(self.model.get_weights())
        print(self.model.get_weights())

        # Show models details
        self.model.summary()
        target_dqn_model.summary()

        # Initializes the reverb's dataset
        self.dataset = make_reverb_dataset(
            server_address=self._db_server,
            table="experience",
            batch_size=batch_size,
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.train_steps = train_steps
        wandb.config.batch_size = batch_size
        wandb.config.learning_rate = learning_rate
        wandb.config.global_clipnorm = global_clipnorm
        wandb.config.gamma = gamma
        wandb.config.tau = tau

    def run(self):
        self.model.fit(
            self.dataset,
            epochs=self._train_steps,
            steps_per_epoch=1,
            verbose=0,
            callbacks=[
                AgentCallback(self._db_server),
                WandbCallback(save_model=False),
                LearningRateScheduler(
                    cosine_schedule(
                        base_lr=wandb.config.learning_rate,
                        total_steps=self._train_steps,
                        warmup_steps=self._warmup_steps,
                    )
                ),
                PrintLR(),
            ],
        )

    def close(self):
        super(Learner, self).close()

        # create the checkpoint of the database
        client = reverb.Client(self._db_server)
        client.checkpoint()
