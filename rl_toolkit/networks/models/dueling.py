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
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Lambda,
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


class TargetModelWrapper:
    def __init__(self, target_dqn_model):
        self._target_dqn_model = target_dqn_model

    def __call__(self, inputs, training):
        return self._target_dqn_model(inputs, training=training)

    @property
    def variables(self):
        return self._target_dqn_model.variables


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
        gamma,
        tau,
        target_dqn_model=None,
        **kwargs
    ):
        super(DuelingDQN, self).__init__(**kwargs)
        self._target_dqn_model_wrapper = (
            TargetModelWrapper(target_dqn_model)
            if target_dqn_model is not None
            else None
        )
        self.gamma = gamma
        self.tau = tau

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

        # Encoder
        self.e_layers = [
            Encoder(embed_dim, ff_mult, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ]

	# Reduce
        # self.flatten = Lambda(lambda x: x[:, -1])
        # self.flatten = GlobalMaxPooling1D()
        self.flatten = GlobalAveragePooling1D()
        
        # Output
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


        # Reduce block
        x = self.flatten(x, training=training)
        # x = self.drop_out(x, training=training)

        # compute value & advantage
        V = self.V(x, training=training)
        A = self.A(x, training=training)

        # advantages have zero mean
        A -= tf.reduce_mean(A, axis=-1, keepdims=True)  # [B, A]

        return V + A  # [B, A]

    def get_action(self, state, temperature):
        return tf.random.categorical(self(state, training=False) / temperature, 1)[0, 0]

    def _update_target(self):
        for source_weight, target_weight in zip(
            self.variables, self._target_dqn_model_wrapper.variables
        ):
            target_weight.assign(
                self.tau * source_weight + (1.0 - self.tau) * target_weight
            )

    def train_step(self, sample):
        # Set dtype
        ext_reward = tf.cast(sample.data["ext_reward"], dtype=self.dtype)
        terminal = tf.cast(sample.data["terminal"], dtype=self.dtype)

        # predict next Q
        next_Q = self._target_dqn_model_wrapper(
            sample.data["next_observation"], training=False
        )
        next_Q = tf.reduce_max(next_Q, axis=-1)

        # get targets
        targets = self(sample.data["observation"])
        indices = tf.range(tf.shape(targets)[0], dtype=sample.data["action"].dtype)
        indices = tf.transpose([indices, sample.data["action"]])
        updates = ext_reward[:, -1] + (1.0 - terminal[:, -1]) * self.gamma * next_Q
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
        tf.debugging.Assert(
            tf.math.less(dqn_loss, 100.0), ["The loss is exploding!!! Value:", dqn_loss]
        )

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(dqn_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------- Soft update target networks -------------------- #
        self._update_target()

        return {
            "dqn_loss": dqn_loss,
        }
