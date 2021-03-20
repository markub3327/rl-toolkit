import tensorflow as tf


class Linear(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear learning rate scheduler
    ==============================

    Args:
        initial_value: initialization value of learning rate (float)

    Returns:
        The return learning rate.
    """

    def __init__(self, initial_value, decay):
        super(Linear, self).__init__()

        self.decay = decay
        self.init_lr = initial_value

    def __call__(self, step):
        return self.init_lr * 1.0 / (1.0 + self.decay * step)