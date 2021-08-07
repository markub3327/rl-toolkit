import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, state_dim, action_dim, max_size):
        self.state_buffer = np.zeros((max_size,) + state_dim, dtype=np.float32)
        self.action_buffer = np.zeros((max_size,) + action_dim, dtype=np.float32)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, state, action, reward, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size - 1, size=batch_size)

        return (
            self.state_buffer[idxs],
            self.state_buffer[idxs + 1],
            self.action_buffer[idxs],
            self.reward_buffer[idxs],
            self.done_buffer[idxs],
        )
