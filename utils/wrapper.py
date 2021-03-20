import gym
import numpy as np

class TimestepsWrapper(gym.Wrapper):
    def __init__(self, env, memory_size=64):
        super().__init__(env)
        self.env = env
        self.memory_size = memory_size
        self.t = 0   # init timestep

        # short memory (bad solution because data are not compressed)
        # Auto-Encoder may be better ????
        self.obs_buf = np.zeros((memory_size,) + self.env.observation_space.shape, dtype=np.float32)

    def reset(self):
        # clear memory at start of episode (Zeros initializer)
        self.obs_buf.fill(0.0)
        self.t = 0

        # init env
        self.env.reset()

        #print(f'{self.t}: {self.obs_buf.flatten()}')

        return self.obs_buf.flatten()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        self.obs_buf[self.t] = next_state
        self.t = (self.t + 1) % self.memory_size

        #print(f'{self.t}: {self.obs_buf.flatten()}')

        return self.obs_buf.flatten(), reward, done, info