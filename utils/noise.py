from abc import ABC, abstractmethod
import tensorflow as tf


class ActionNoise(ABC):
    def __init__(self, shape):
        self.shape = tf.constant(shape)

    @abstractmethod
    def sample(self):
        ...
    
    @abstractmethod
    def reset(self):
        ...

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    :param mean: the mean of the noise (float)
    :param sigma: the scale of the noise (float)
    :param shape: shape of generated noise matrix
    :param theta: the rate of mean reversion (float)
    :param dt: the timestep for the noise (float)

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """

    def __init__(self, mean, sigma, shape, theta=0.15, dt=1e-2):
        self._theta = tf.constant(theta)
        self._mu = tf.constant(mean)
        self._sigma = tf.constant(sigma)
        self._dt = tf.constant(dt)
        self.noise_prev = tf.Variable(tf.zeros(self.shape), dtype=tf.float32)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__(shape)

    @tf.function
    def sample(self):
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * tf.math.sqrt(self._dt) * tf.random.normal(self.shape)
        )
        self.noise_prev.assign(noise)
        return noise

    @tf.function
    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev.assign(tf.zeros(self.shape))


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise
    :param mean: the mean value of the noise (float)
    :param sigma: the scale of the noise (float)
    :param shape: shape of generated noise matrix
    """

    def __init__(self, mean, sigma, shape):
        self._mu = tf.constant(mean)
        self._sigma = tf.constant(sigma)
        super(NormalActionNoise, self).__init__(shape)

    @tf.function
    def sample(self):
        return tf.random.normal(self.shape, mean=self._mu, stddev=self._sigma)

    def reset(self):
        """
        Not using with normal distribution
        """
        pass
