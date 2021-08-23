import reverb
import tensorflow as tf


class VariableContainer:
    def __init__(
        self,
        # ---
        db_server: str,
        # ---
        table_name: str,
        variables: dict,
    ):
        self._table_name = table_name
        self._variables = variables

        # Initializes the reverb client
        self.tf_client = reverb.TFClient(server_address=f"{db_server}:8000")

        # variables signature for variable container table
        self.signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self._variables,
        )
        self.dtypes = tf.nest.map_structure(lambda spec: spec.dtype, self.signature)

    def update_variables(self):
        sample = self.tf_client.sample(
            self._table_name, data_dtypes=[self.dtypes]
        ).data[0]
        for variable, value in zip(
            tf.nest.flatten(self._variables), tf.nest.flatten(sample)
        ):
            variable.assign(value)

    def push_variables(self):
        self.tf_client.insert(
            data=tf.nest.flatten(self._variables),
            tables=tf.constant([self._table_name]),
            priorities=tf.constant([1.0], dtype=tf.float64),
        )

    def __getitem__(self, key):
        return self._variables[key]
