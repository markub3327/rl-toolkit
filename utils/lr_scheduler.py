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

    def __init__(self, initial_value, max_step):
        super(Linear, self).__init__()

        self.initial_value = initial_value
        self.max_step = max_step

    def __call__(self, step):
        lr = (1.0 - (step / self.max_step)) * self.initial_value
        tf.print(lr)
        return lr
