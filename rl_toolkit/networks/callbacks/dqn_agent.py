import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from rl_toolkit.utils import VariableContainer


class DQNAgentCallback(Callback):
    def __init__(self, db_server: str):
        super(DQNAgentCallback, self).__init__()
        self._db_server = db_server

    def on_train_begin(self, logs=None):
        # Variables
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.uint64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )
        self._stop_agents = tf.Variable(
            False,
            trainable=False,
            dtype=tf.bool,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        # Table for storing variables
        self._variable_container = VariableContainer(
            db_server=self._db_server,
            table="variables",
            variables={
                "policy_variables": self.model.variables,
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
            },
        )

        # Init variable container from DB server
        self._variable_container.update_variables()

    def on_epoch_end(self, epoch, logs=None):
        self._train_step.assign_add(1)
        self._variable_container.push_variables()

    def on_train_end(self, logs=None):
        self._stop_agents.assign(True)
        self._variable_container.push_variables()
