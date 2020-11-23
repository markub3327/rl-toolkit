import numpy as np
from typing import Optional

class OrnsteinUhlenbeckActionNoise(object):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean,
        sigma,
        size,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._size = size
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros(self._size)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._size)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._size)

class NormalActionNoise(object):
    """
    A Gaussian action noise
    :param mean: the mean value of the noise
    :param sigma: the scale of the noise (std here)
    """

    def __init__(self, mean, sigma, size):
        self._mu = mean
        self._sigma = sigma
        self._size = size
        super(NormalActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, size=self._size)

    def reset(self):
        """
        Not using with normal distribution
        """
        pass