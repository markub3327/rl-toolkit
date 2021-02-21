import tensorflow as tf


def linear(t, initial_value):
    """
    Linear learning rate scheduler
    :param t: timestep of training process (float)
    :param initial_value: initialization value of learning rate (float)
    :return: (float)
    """
    return (1.0 - t) * initial_value