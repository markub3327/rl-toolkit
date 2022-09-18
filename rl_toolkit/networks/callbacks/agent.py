import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from rl_toolkit.utils import VariableContainer


class AgentCallback(Callback):
    def __init__(self, db_server: str):
        super(AgentCallback, self).__init__()

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
            table="variable",
            variables={
                "train_step": self._train_step,
                "stop_agents": self._stop_agents,
                "policy_variables": self.model.actor.variables,
            },
        )

        # Init variable container from DB server
        self._variable_container.update_variables()

    def on_epoch_end(self, epoch, logs=None):
        # increase the training step
        self._train_step.assign_add(1)

        # Store new actor's params
        self._variable_container.push_variables()

    def on_train_end(self, logs=None):
        # Stop the agents
        self._stop_agents.assign(True)
        self._variable_container.push_variables()
