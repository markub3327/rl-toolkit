import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Activation, Add, Dense


class Counter(Model):
    """
    Counter
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        gamma (float): the discount factor

    References:
        - [DORA The Explorer: Directed Outreaching Reinforcement Action-Selection](https://arxiv.org/abs/1804.04012)
    """

    def __init__(
        self, units: list, gamma: float, target_model: Model, tau: float, **kwargs
    ):
        super(Counter, self).__init__(**kwargs)

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(tau)
        self._target_model = target_model

        # 1. layer
        self.fc_0 = Dense(
            units=units[0],
            activation="relu",
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )

        # 2. layer     TODO(markub3327): Transformer
        self.fc_1 = Dense(
            units=units[1],
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )
        self.fc_2 = Dense(
            units=units[1],
            kernel_initializer=VarianceScaling(
                distribution="uniform", mode="fan_in", scale=1.0
            ),
        )
        self.add_0 = Add()
        self.activ_0 = Activation("relu")

        # Output layer
        self.e_value = Dense(
            1,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="e_value",
        )
        self.activ_1 = Activation("sigmoid")

        if target_model is not None:
            self._update_target(self, self._target_model, tau=1.0)

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.variables, net_targ.variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def call(self, inputs):
        # 1. layer
        state = self.fc_0(inputs[0])

        # 2. layer
        state = self.fc_1(state)
        action = self.fc_2(inputs[1])
        x = self.add_0([state, action])
        x = self.activ_0(x)

        # Output layer
        e_value = self.e_value(x)

        # Intrinsic Reward
        counter = 1.0 / tf.math.sqrt(-tf.math.log_sigmoid(e_value))

        # E-value
        e_value = self.activ_1(e_value)

        return counter, e_value

    def train_step(self, sample):
        # Get trainable variables
        counter_variables = self.trainable_variables

        # -------------------- (SARSA method) -------------------- #
        next_e_value = self._target_model(
            [
                sample.data["next_observation"],
                sample.data["next_action"],
            ]
        )
        target_e_value = tf.stop_gradient(
            (1.0 - tf.cast(sample.data["terminal"], dtype=tf.float32))
            * self.gamma
            * next_e_value
        )

        with tf.GradientTape() as tape:
            _, e_value = self([sample.data["observation"], sample.data["action"]])
            counter_loss = tf.nn.compute_average_loss(
                tf.keras.losses.log_cosh(target_e_value, e_value)
            )

        # Compute gradients
        counter_gradients = tape.gradient(counter_loss, counter_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(counter_gradients, counter_variables))

        # -------------------- Soft update target networks -------------------- #
        self._update_target(self, self._target_model, tau=self.tau)

        return {
            "counter_loss": counter_loss,
        }
