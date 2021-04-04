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

    def __init__(self, initial_value, max_steps, warmup_steps):
        super(Linear, self).__init__()

        self.max_steps = max_steps
        self.init_lr = initial_value
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return (1.0 - ((step + self.warmup_steps) / self.max_steps)) * self.init_lr
