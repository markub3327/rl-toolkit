import reverb

import tensorflow as tf


class ReverbPolicyContainer:
    def __init__(self, server_name, actor):

        # actual training step
        self.train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        self.vars = {
            "train_step": self.train_step,
            "actor_variables": actor.variables,
        }
        self._dtypes = tf.nest.map_structure(lambda spec: spec.dtype, self.vars)

        self._tf_client = reverb.TFClient(server_address=f"{server_name}:8000")

    def insert(self, train_step):
        self.train_step.assign(train_step)
        self._tf_client.insert(
            data=tf.nest.flatten(self.vars),
            tables=tf.constant(["model_vars"]),
            priorities=tf.constant([1.0], dtype=tf.float64),
        )

    def update(self):
        sample = self._tf_client.sample("model_vars", data_dtypes=[self._dtypes])
        for variable, value in zip(
            tf.nest.flatten(self.vars), tf.nest.flatten(sample.data[0])
        ):
            variable.assign(value)
