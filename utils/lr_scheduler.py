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

        self.max_step = max_step
        self.init_lr = initial_value

    def __call__(self, step):
        return (1.0 - (step / self.max_step)) * self.initial_value