import reverb

import tensorflow as tf


class ReverbSyncPolicy:
    def __init__(self, actor):

        # actual training step
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        self.vars = {
            "train_step": self._train_step,
            "actor_variables": actor.variables,
        }
        variable_container_signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self.vars,
        )
        print(f"Signature of variables: \n{variable_container_signature}")

        self._dtypes = tf.nest.map_structure(lambda spec: spec.dtype, self.vars)

        self._tf_client = reverb.TFClient(server_address="localhost:8000")

    def update(self, train_step):
        self._train_step.assign(train_step)
        self._tf_client.insert(
            data=tf.nest.flatten(self.vars),
            tables="model_vars",
            priorities=tf.constant([1.0], dtype=tf.float64),
        )

    def sync(self):
        sample = self._tf_client.sample("model_vars", data_dtypes=[self._dtypes])
        for variable, value in zip(
            tf.nest.flatten(self.vars), tf.nest.flatten(sample.data[0])
        ):
            variable.assign(value)
