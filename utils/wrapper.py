import gym
import numpy as np

class TimestepsWrapper(gym.Wrapper):
    def __init__(self, env, memory_size):
        super().__init__(env)
        self.env = env
        self.memory_size = memory_size

        # short memory (bad solution because data are not compressed)
        # Auto-Encoder may be better ????

    def reset(self):
        # init env
        obs = self.env.reset()

        # clear memory at start of episode
        self.obs_buf = np.full((self.memory_size,) + self.env.observation_space.shape, obs, dtype=np.float32)

        #print(f'{self.obs_buf.shape}')

        return self.obs_buf.flatten()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        self.obs_buf = np.concatenate((self.obs_buf[1:], next_state), axis=0)

        #print(f'{self.obs_buf.flatten()}')

        return self.obs_buf.flatten(), reward, done, info