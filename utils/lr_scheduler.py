import tensorflow as tf


def linear(t, initial_value):
    """
    Linear learning rate scheduler
    ==============================

    Args:
        t: timestep of training process (float)
        initial_value: initialization value of learning rate (float)

    Returns:
        The return learning rate.
    """
    return (1.0 - t) * initial_value
