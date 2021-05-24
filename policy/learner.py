import reverb

# import wandb

import tensorflow as tf

from .network import Actor, Critic
from .reverb_utils import ReverbSyncPolicy


class Learner:
    """
    Agent
    =================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        buffer_size (int): maximum size of the replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): learning rate for actor's optimizer
        critic_learning_rate (float): learning rate for critic's optimizer
        alpha_learning_rate (float): learning rate for alpha's optimizer
        learning_starts (int): number of interactions before using policy network
    """

    def __init__(
        self,
        env,
        # ---
        max_steps: int,
        # ---
        learning_starts: int = int(1e4),
        # ---
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        # ---
        actor_learning_rate: float = 7.3e-4,
        critic_learning_rate: float = 7.3e-4,
        alpha_learning_rate: float = 7.3e-4,
        # ---
        tau: float = 0.01,
        gamma: float = 0.99,
        # ---
        model_a_path: str = None,
        model_c1_path: str = None,
        model_c2_path: str = None,
    ):
        self._max_steps = max_steps
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)

        # logging metrics
        self._loss_a = tf.keras.metrics.Mean("loss_a", dtype=tf.float32)
        self._loss_c1 = tf.keras.metrics.Mean("loss_c1", dtype=tf.float32)
        self._loss_c2 = tf.keras.metrics.Mean("loss_c2", dtype=tf.float32)
        self._loss_alpha = tf.keras.metrics.Mean("loss_alpha", dtype=tf.float32)

        # ---------------- Init param 'alpha' (Lagrangian constraint) ---------------- #
        self._log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self._alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self._alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate, name="alpha_optimizer"
        )
        self._target_entropy = tf.cast(
            -tf.reduce_prod(env.action_space.shape), dtype=tf.float32
        )

        # Actor network
        self._actor = Actor(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            learning_rate=actor_learning_rate,
            model_path=model_a_path,
        )

        # Critic network & target network
        self._critic_1 = Critic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c1_path,
        )
        self._critic_targ_1 = Critic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c1_path,
        )

        # Critic network & target network
        self._critic_2 = Critic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c2_path,
        )
        self._critic_targ_2 = Critic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c2_path,
        )

        # first make a hard copy
        self._update_target(self._critic_1, self._critic_targ_1, tau=tf.constant(1.0))
        self._update_target(self._critic_2, self._critic_targ_2, tau=tf.constant(1.0))

        # Initialize the Reverb server
        self._db = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="uniform_table",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(learning_starts),
                    max_size=buffer_size,
                    signature={
                        "obs": tf.TensorSpec(
                            [*env.observation_space.shape],
                            dtype=env.observation_space.dtype,
                        ),
                        "action": tf.TensorSpec(
                            [*env.action_space.shape], dtype=env.action_space.dtype
                        ),
                        "reward": tf.TensorSpec([1], dtype=tf.float32),
                        "obs2": tf.TensorSpec(
                            [*env.observation_space.shape],
                            dtype=env.observation_space.dtype,
                        ),
                        "done": tf.TensorSpec([1], dtype=tf.float32),
                    },
                ),
                reverb.Table(  # Actor's variables
                    name="model_vars",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature={
                        "train_step": tf.TensorSpec(
                            [],
                            dtype=tf.int32,
                        ),
                        "actor_variables": tf.nest.map_structure(
                            lambda variable: tf.TensorSpec(
                                variable.shape, dtype=variable.dtype
                            ),
                            self._actor.model.variables,
                        ),
                    },
                ),
            ],
            port=8000,
        )

        # Dataset samples sequences of length 3 and streams the timesteps one by one.
        # This allows streaming large sequences that do not necessarily fit in memory.
        self._dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address="localhost:8000",
            table="uniform_table",
            max_in_flight_samples_per_worker=10,
        ).batch(batch_size)

        self.reverb_sync_policy = ReverbSyncPolicy(self._actor.model)
        self.reverb_sync_policy.update(0)

        # init Weights & Biases
        # wandb.init(project="rl-toolkit")

        # set Weights & Biases
        # wandb.config.max_steps = max_steps
        # wandb.config.env_steps = env_steps
        # wandb.config.gradient_steps = gradient_steps
        # wandb.config.learning_starts = learning_starts
        # wandb.config.buffer_size = buffer_size
        # wandb.config.batch_size = batch_size
        # wandb.config.actor_learning_rate = actor_learning_rate
        # wandb.config.critic_learning_rate = critic_learning_rate
        # wandb.config.alpha_learning_rate = alpha_learning_rate
        # wandb.config.tau = tau
        # wandb.config.gamma = gamma

    @tf.function
    def run(self):
        for step in tf.range(self._max_steps):
            # iterate over dataset
            for sample in self._dataset:
                # re-new noise matrix every update of 'log_std' params
                self._actor.reset_noise()

                # Alpha param update
                self._loss_alpha.update_state(self._update_alpha(sample))

                l_c1, l_c2 = self._update_critic(sample)
                self._loss_c1.update_state(l_c1)
                self._loss_c2.update_state(l_c2)

                # Actor model update
                self._loss_a.update_state(self._update_actor(sample))

                # ------------------- soft update target networks ------------------- #
                self._update_target(self._critic_1, self._critic_targ_1, tau=self._tau)
                self._update_target(self._critic_2, self._critic_targ_2, tau=self._tau)

                tf.print("=============================================")
                tf.print(f"Step: {step}")
                tf.print(f"Alpha: {self._alpha}")
                tf.print(f"Actor's loss: {self._loss_a.result()}")
                tf.print(f"Critic 1's loss: {self._loss_c1.result()}")
                tf.print(f"Critic 2's loss: {self._loss_c2.result()}")
                tf.print(f"Alpha's loss: {self._loss_alpha.result()}")
                tf.print("=============================================")
                tf.print(f"Training ... {(step * 100) / self._max_steps} %")

                # log to W&B
            #                wandb.log(
            #                    {
            #                        "loss_a": self._loss_a.result(),
            #                        "loss_c1": self._loss_c1.result(),
            #                        "loss_c2": self._loss_c2.result(),
            #                        "loss_alpha": self._loss_alpha.result(),
            #                        "alpha": self._alpha,
            #                    },
            #                    step=step,
            #                )

            # save params to table
            self.reverb_sync_policy.update(step)

            # reset logger
            self._loss_a.reset_states()
            self._loss_c1.reset_states()
            self._loss_c2.reset_states()
            self._loss_alpha.reset_states()

    def save(self, path):
        # Save model to local drive
        self._actor.model.save(f"{path}model_A.h5")
        self._critic_1.model.save(f"{path}model_C1.h5")
        self._critic_2.model.save(f"{path}model_C2.h5")

    # -------------------------------- update alpha ------------------------------- #
    def _update_alpha(self, batch):
        _, log_pi = self._actor.predict(batch.data["obs"])

        self._alpha.assign(tf.exp(self._log_alpha))
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._log_alpha * tf.stop_gradient(log_pi + self._target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    # -------------------------------- update critic ------------------------------- #
    def _update_critic(self, batch):
        next_action, next_log_pi = self._actor.predict(batch.data["obs2"])

        # target Q-values
        next_q_1 = self._critic_targ_1.model([batch.data["obs2"], next_action])
        next_q_2 = self._critic_targ_2.model([batch.data["obs2"], next_action])
        next_q = tf.minimum(next_q_1, next_q_2)

        # Bellman Equation
        Q_targets = tf.stop_gradient(
            batch.data["reward"]
            + (1 - batch.data["done"])
            * self._gamma
            * (next_q - self._alpha * next_log_pi)
        )

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self._critic_1.model([batch.data["obs"], batch.data["action"]])
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=q_values
            )
            q1_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q1_loss, self._critic_1.model.trainable_variables)
        self._critic_1.optimizer.apply_gradients(
            zip(grads, self._critic_1.model.trainable_variables)
        )

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self._critic_2.model([batch.data["obs"], batch.data["action"]])
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=q_values
            )
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self._critic_2.model.trainable_variables)
        self._critic_2.optimizer.apply_gradients(
            zip(grads, self._critic_2.model.trainable_variables)
        )

        return q1_loss, q2_loss

    # -------------------------------- update actor ------------------------------- #
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_pi = self._actor.predict(batch.data["obs"])

            # predict q value
            q_1 = self._critic_1.model([batch.data["obs"], y_pred])
            q_2 = self._critic_2.model([batch.data["obs"], y_pred])
            q = tf.minimum(q_1, q_2)

            a_losses = self._alpha * log_pi - q
            a_loss = tf.nn.compute_average_loss(a_losses)

        grads = tape.gradient(a_loss, self._actor.model.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.model.trainable_variables)
        )

        return a_loss

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
