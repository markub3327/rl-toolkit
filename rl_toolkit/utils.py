import tensorflow as tf

import reverb


class VariableContainer:
    def __init__(
        self,
        # ---
        db_server: str,
        # ---
        actor,
    ):

        # Initializes the reverb client
        self.tf_client = reverb.TFClient(server_address=f"{db_server}:8000")

        # actual training step
        self.train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        self.stop_agents = tf.Variable(
            False,
            trainable=False,
            dtype=tf.bool,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        # prepare variable container
        self._variables_container = {
            "train_step": self.train_step,
            "stop_agents": self.stop_agents,
            "policy_variables": actor.variables,
        }

        # variables signature for variable container table
        self.variable_container_signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self._variables_container,
        )
        self.dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, self.variable_container_signature
        )

    @tf.function
    def update_variables(self):
        sample = self.tf_client.sample("variables", data_dtypes=[self.dtypes])
        for variable, value in zip(
            tf.nest.flatten(self._variables_container), tf.nest.flatten(sample.data[0])
        ):
            variable.assign(value)

    def push_variables(self):
        self.tf_client.insert(
            data=tf.nest.flatten(self._variables_container),
            tables=tf.constant(["variables"]),
            priorities=tf.constant([1.0], dtype=tf.float64),
        )
