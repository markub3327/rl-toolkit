from rl_toolkit.networks import Actor, Critic
from rl_toolkit.types import Transition

import os
import math
import reverb
import wandb

import tensorflow as tf


class Learner:
    """
    Learner (based on Soft Actor-Critic)
    =================

    Paper: https://arxiv.org/pdf/1812.05905.pdf

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        gradient_steps (int): number of update steps after each rollout
        learning_starts (int): number of interactions before using policy network
        buffer_capacity (int): the capacity of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): learning rate for actor's optimizer
        critic_learning_rate (float): learning rate for critic's optimizer
        alpha_learning_rate (float): learning rate for alpha's optimizer
        tau (float): the soft update coefficient for target networks
        gamma (float): the discount factor
        model_a_path (str): path to the actor's model
        model_c1_path (str): path to the critic_1's model
        model_c2_path (str): path to the critic_2's model
        save_path (str): path to the models for saving
        db_path (str): path to the database
        logging_wandb (bool): logging by WanDB
        log_freq (int): logging frequency
    """

    def __init__(
        self,
        # ---
        env,
        max_steps: int,
        gradient_steps: int = 64,
        learning_starts: int = 10000,
        # ---
        buffer_capacity: int = 1000000,
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
        save_path: str = None,
        db_path: str = None,
        # ---
        logging_wandb: bool = False,
        log_freq: int = 64,
    ):
        self._env = env
        self._max_steps = max_steps
        self._learning_starts = learning_starts
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._save_path = save_path
        self._logging_wandb = logging_wandb
        self._log_freq = log_freq

        # init param 'alpha' - Lagrangian constraint
        self._log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self._alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self._alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate, name="alpha_optimizer"
        )
        self._target_entropy = tf.cast(
            -tf.reduce_prod(self._env.action_space.shape), dtype=tf.float32
        )

        # Actor network (for learner)
        self._actor = Actor(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=actor_learning_rate,
            model_path=model_a_path,
        )

        # Critic network & target network
        self._critic_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c1_path,
        )
        self._critic_targ_1 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c1_path,
        )

        # Critic network & target network
        self._critic_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c2_path,
        )
        self._critic_targ_2 = Critic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=model_c2_path,
        )

        # first make a hard copy
        self._update_target(self._critic_1, self._critic_targ_1, tau=tf.constant(1.0))
        self._update_target(self._critic_2, self._critic_targ_2, tau=tf.constant(1.0))

        if db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=db_path)

        # prepare variable container
        self._variables_container = {
            "policy_variables": self._actor.model.variables,
        }

        # variables signature for variable container table
        variable_container_signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self._variables_container,
        )

        # Ratio for samples per insert rate limiting tolerance
        SAMPLES_PER_INSERT_TOLERANCE_RATIO = 0.1

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experience",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                        min_size_to_sample=learning_starts,
                        samples_per_insert=gradient_steps,
                        error_buffer=(
                            learning_starts
                            * SAMPLES_PER_INSERT_TOLERANCE_RATIO
                            * gradient_steps
                        ),
                    ),
                    max_size=buffer_capacity,
                    max_times_sampled=0,
                    signature={
                        "observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "action": tf.TensorSpec(
                            [*self._env.action_space.shape],
                            self._env.action_space.dtype,
                        ),
                        "reward": tf.TensorSpec([1], tf.float32),
                        "next_observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "terminal": tf.TensorSpec([1], tf.float32),
                    },
                ),
                reverb.Table(  # Variable container
                    name="variables",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature=variable_container_signature,
                ),
            ],
            port=8000,
            checkpointer=checkpointer,
        )

        # Initializes the reverb client
        self.client = reverb.Client("localhost:8000")
        self.tf_client = reverb.TFClient(server_address="localhost:8000")
        self._dataset_iterator = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address="localhost:8000",
                table="experience",
                max_in_flight_samples_per_worker=10,
            ).batch(batch_size)
        )

        # init Weights & Biases
        if self._logging_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.gradient_steps = gradient_steps
            wandb.config.learning_starts = learning_starts
            wandb.config.buffer_capacity = buffer_capacity
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.alpha_learning_rate = alpha_learning_rate
            wandb.config.tau = tau
            wandb.config.gamma = gamma

        # init actor's params in DB
        self._push_variables()

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def _push_variables(self):
        self.tf_client.insert(
            data=tf.nest.flatten(self._variables_container),
            tables=tf.constant(["variables"]),
            priorities=tf.constant([1.0], dtype=tf.float64),
        )

    # -------------------------------- update critic ------------------------------- #
    def _update_critic(self, batch):
        next_action, next_log_pi = self._actor.predict(batch.next_observation)

        # target Q-values
        next_q_1 = self._critic_targ_1.model([batch.next_observation, next_action])
        next_q_2 = self._critic_targ_2.model([batch.next_observation, next_action])
        next_q = tf.minimum(next_q_1, next_q_2)
        # tf.print(f'nextQ: {next_q.shape}')

        # Bellman Equation
        Q_targets = tf.stop_gradient(
            batch.reward
            + (1.0 - batch.terminal)
            * self._gamma
            * (next_q - self._alpha * next_log_pi)
        )
        # tf.print(f'qTarget: {Q_targets.shape}')

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self._critic_1.model([batch.observation, batch.action])
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_targets, y_pred=q_values
            )
            q1_loss = tf.nn.compute_average_loss(q_losses)
        #    tf.print(f'q_val: {q_values.shape}')

        grads = tape.gradient(q1_loss, self._critic_1.model.trainable_variables)
        self._critic_1.optimizer.apply_gradients(
            zip(grads, self._critic_1.model.trainable_variables)
        )

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self._critic_2.model([batch.observation, batch.action])
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
            y_pred, log_pi = self._actor.predict(batch.observation)
            # tf.print(f'log_pi: {log_pi.shape}')

            # predict q value
            q_1 = self._critic_1.model([batch.observation, y_pred])
            q_2 = self._critic_2.model([batch.observation, y_pred])
            q = tf.minimum(q_1, q_2)
            # tf.print(f'q: {q.shape}')

            a_losses = self._alpha * log_pi - q
            a_loss = tf.nn.compute_average_loss(a_losses)
            # tf.print(f'a_losses: {a_losses}')

        grads = tape.gradient(a_loss, self._actor.model.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.model.trainable_variables)
        )

        return a_loss

    # -------------------------------- update alpha ------------------------------- #
    def _update_alpha(self, batch):
        _, log_pi = self._actor.predict(batch.observation)
        # tf.print(f'y_pred: {y_pred.shape}')
        # tf.print(f'log_pi: {log_pi.shape}')

        self._alpha.assign(tf.exp(self._log_alpha))
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._log_alpha * tf.stop_gradient(log_pi + self._target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        #    tf.print(f'alpha_losses: {alpha_losses.shape}')

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    @tf.function
    def _update(self):
        # Get data from replay buffer
        sample = next(self._iterator)
        transitions: Transition = sample.data

        # re-new noise matrix every update of 'log_std' params
        self._actor.reset_noise()

        # Alpha param update
        loss_alpha = self._update_alpha(transitions)

        # Critic model update
        loss_c1, loss_c2 = self._update_critic(transitions)

        # Actor model update
        loss_a = self._update_actor(transitions)

        # -------------------- soft update target networks -------------------- #
        self._update_target(self._critic_1, self._critic_targ_1, tau=self._tau)
        self._update_target(self._critic_2, self._critic_targ_2, tau=self._tau)

        # store new actor's params
        self._push_variables()

        return loss_alpha, loss_c1, loss_c2, loss_a

    def run(self):
        for step in range(self._learning_starts, self._max_steps, 1):
            # update models
            loss_alpha, loss_c1, loss_c2, loss_a = self._update()

            # log metrics
            self._logging_models(loss_alpha, loss_c1, loss_c2, loss_a, step)

    def _logging_models(self, loss_alpha, loss_c1, loss_c2, loss_a, step):
        # log into console
        if step % self._log_freq == 0:
            print("=============================================")
            print(f"Step: {step}")
            print(f"Actor's loss: {loss_a}")
            print(f"Critic 1's loss: {loss_c1}")
            print(f"Critic 2's loss: {loss_c2}")
            print("=============================================")
            print(
                f"Training ... {math.floor(step * 100.0 / self._max_steps)} %"
            )  # noqa

        # log into wandb cloud
        if self._logging_wandb:
            wandb.log(
                {
                    "loss_a": loss_a,
                    "loss_c1": loss_c1,
                    "loss_c2": loss_c2,
                    "loss_alpha": loss_alpha,
                    "alpha": self._alpha,
                },
                step=step,
            )
        self._clear_metrics()  # clear stored metrics of losses

    def _clear_metrics(self):
        # reset logger
        self._loss_a.reset_states()
        self._loss_c1.reset_states()
        self._loss_c2.reset_states()
        self._loss_alpha.reset_states()

    def save(self):
        if self._save_path is not None:
            # store models
            self._actor.model.save(os.path.join(self._save_path, "model_A.h5"))
            self._critic_1.model.save(os.path.join(self._save_path, "model_C1.h5"))
            self._critic_2.model.save(os.path.join(self._save_path, "model_C2.h5"))

        # store checkpoint of DB
        checkpoint_path = self.client.checkpoint()
        print(checkpoint_path)

    def convert(self):
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(self._actor.model)
        tflite_model = converter.convert()

        # Save the model.
        with open("model_A.tflite", "wb") as f:
            f.write(tflite_model)
