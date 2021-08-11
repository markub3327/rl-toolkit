import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_buf = np.zeros((max_size,) + obs_dim, dtype=np.float32)
        self.obs2_buf = np.zeros((max_size,) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((max_size,) + act_dim, dtype=np.float32)
        self.rew_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            observation=self.obs_buf[idxs],
            next_observation=self.obs2_buf[idxs],
            action=self.act_buf[idxs],
            reward=self.rew_buf[idxs],
            terminal=self.done_buf[idxs],
        )
