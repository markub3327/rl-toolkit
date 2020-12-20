import tensorflow as tf


class OrnsteinUhlenbeckActionNoise:
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """

    def __init__(self, mean, sigma, size, theta=0.15, dt=1e-2):
        self._theta = tf.constant(theta)
        self._mu = tf.constant(mean)
        self._sigma = tf.constant(sigma)
        self._dt = tf.constant(dt)
        self._size = tf.constant(size)
        self.noise_prev = tf.Variable(tf.zeros(self._size), dtype=tf.float32)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__()

    @tf.function
    def sample(self):
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * tf.math.sqrt(self._dt) * tf.random.normal(self._size))
        self.noise_prev.assign(noise)
        return noise

    @tf.function
    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev.assign(tf.zeros(self._size))

class NormalActionNoise:
    """
    A Gaussian action noise
    :param mean: the mean value of the noise
    :param sigma: the scale of the noise (std here)
    """

    def __init__(self, mean, sigma, size):
        self._mu = tf.constant(mean)
        self._sigma = tf.constant(sigma)
        self._size = tf.constant(size)
        super(NormalActionNoise, self).__init__()

    @tf.function
    def sample(self):
        return tf.random.normal(self._size, mean=self._mu, stddev=self._sigma)

    @tf.function
    def reset(self):
        """
        Not using with normal distribution
        """
        pass
