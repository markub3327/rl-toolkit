from tensorflow.keras.callbacks import Callback

from rl_toolkit.utils import VariableContainer


class GANCallback(Callback):
    def __init__(self, db_server: str, client: bool = False):
        super(GANCallback, self).__init__()

        self._db_server = db_server
        self._client = client

    def on_train_begin(self, logs=None):
        # Table for storing variables
        self._variable_container = VariableContainer(
            db_server=self._db_server,
            table="variable2",
            variables={
                "discriminator_variables": self.model.discriminator.variables,
            },
        )

        # Init variable container from DB server
        self._variable_container.update_variables()

    def on_epoch_end(self, epoch, logs=None):
        if not self._client:
            # Store new actor's params
            self._variable_container.push_variables()
        else:
            # Update actor's params
            self._variable_container.update_variables()

    def on_train_end(self, logs=None):
        if not self._client:
            # Stop the agents
            self._variable_container.push_variables()
