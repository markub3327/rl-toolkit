import tensorflow as tf


def clipped_linear(x, clip_value=2.0):
    """
    Clipped linear activation function
    ===============

    Attributes:
        x: input tensor
        clip_value (float): the limit value of output value

    Returns:
        Clipped tensor with the same shape and dtype of input `x`.
    """
    return tf.clip_by_value(x, -clip_value, clip_value)
