from collections import deque

import gymnasium
import numpy as np


class FrameStack(gymnasium.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)

        self.k = k
        self.frames = deque([], maxlen=k)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], k, axis=0)
        self.observation_space = gymnasium.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype,
        )

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._set_ob(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self._set_ob(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _set_ob(self, ob):
        self.frames.append(ob)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.stack(list(self.frames), axis=0)  # stack along time axis
