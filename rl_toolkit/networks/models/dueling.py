import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Orthogonal, TruncatedNormal
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units
        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training):
        x = self.projection(inputs)
        x = x + self.position

        return self.dropout(x, training=training)


class Encoder(Layer):
    def __init__(
        self,
        embed_dim,
        ff_mult,
        num_heads,
        dropout_rate,
        attention_dropout_rate,
        **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )

        self.dense_0 = Dense(
            units=embed_dim * ff_mult,
            activation="gelu",
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )
        self.dense_1 = Dense(
            units=embed_dim, kernel_initializer=TruncatedNormal(stddev=0.02)
        )

        self.dropout_0 = Dropout(rate=dropout_rate)
        self.dropout_1 = Dropout(rate=dropout_rate)

        self.norm_0 = LayerNormalization(epsilon=1e-6)
        self.norm_1 = LayerNormalization(epsilon=1e-6)

        self.add_0 = Add()
        self.add_1 = Add()

    def call(self, inputs, training):
        # Attention block
        x = self.norm_0(inputs)
        x = self.mha(
            query=x,
            value=x,
            key=x,
            training=training,
        )
        x = self.dropout_0(x, training=training)
        x = self.add_0([x, inputs])

        # MLP block
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        y = self.dropout_1(y, training=training)

        return self.add_1([x, y])


class DuelingDQN(Model):
    def __init__(
        self,
        action_space,
        num_layers,
        embed_dim,
        ff_mult,
        num_heads,
        dropout_rate,
        attention_dropout_rate,
        target_dqn_model,
        gamma,
        tau,
        **kwargs
    ):
        super(DuelingDQN, self).__init__(**kwargs)
        self._target_dqn_model = target_dqn_model
        self.gamma = gamma
        self.tau = tau

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

        # Encoder
        self.e_layers = [
            Encoder(embed_dim, ff_mult, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ]

        # Output
        self.norm = LayerNormalization(epsilon=1e-6)
        self.V = Dense(
            1,
            activation=None,
            kernel_initializer=Orthogonal(0.01),
        )
        self.A = Dense(
            action_space,
            activation=None,
            kernel_initializer=Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.pos_embs(inputs, training=training)

        for layer in self.e_layers:
            x = layer(x, training=training)

        x = self.norm(x, training=training)

        # select last timestep for prediction a_t
        x = x[:, -1]

        # compute value & advantage
        V = self.V(x, training=training)
        A = self.A(x, training=training)

        # advantages have zero mean
        A -= tf.reduce_mean(A, axis=-1, keepdims=True)  # [B, A]

        return V + A  # [B, A]

    def get_action(self, state, temperature):
        return tf.random.categorical(self(state, training=False) / temperature, 1)[0, 0]

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.variables, net_targ.variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
            
    def train_step(self, sample):
        # Set dtype
        ext_reward = tf.cast(sample.data["ext_reward"], dtype=self.dtype)
        terminal = tf.cast(sample.data["terminal"], dtype=self.dtype)

        # predict next Q
        next_Q = self._target_dqn_model(sample.data["next_observation"], training=False)
        next_Q = tf.reduce_max(next_Q, axis=-1)

        # get targets
        targets = self(sample.data["observation"])
        indices = tf.range(tf.shape(targets)[0])
        indices = tf.transpose([indices, sample.data["action"]])
        updates = ext_reward + (1.0 - terminal) * self.gamma * next_Q
        targets = tf.stop_gradient(
            tf.tensor_scatter_nd_update(targets, indices, updates)
        )

        #              update DQN              #
        with tf.GradientTape() as tape:
            y_pred = self(sample.data["observation"], training=True)
            dqn_loss = tf.nn.compute_average_loss(
                tf.keras.losses.log_cosh(targets, y_pred)
            )

        # check exploiding loss
        assert dqn_loss < 100, "The loss is exploding"

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(dqn_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------- Soft update target networks -------------------- #
        self._update_target(self, self._target_dqn_model, tau=self.tau)

        return {
            "dqn_loss": dqn_loss,
        }
