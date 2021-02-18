from abc import ABC, abstractmethod

import tensorflow as tf


class OffPolicy(ABC):
    """
    The base for Off-Policy algorithms
    :param tau: the soft update coefficient (float)
    :param gamma: the discount factor (float)
    :param lr_scheduler: type of learning rate scheduler
    """

    def __init__(
        self,
        tau: float,
        gamma: float,
        lr_scheduler
    ):

        self.gamma = tf.constant(gamma)
        self.tau = tf.constant(tau)
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def get_action(self, state):
        ...

    @tf.function
    def update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    @abstractmethod
    def update(self, rpm, epoch, batch_size, gradient_steps, logging_wandb):
        ...