import gymnasium
import numpy as np
from collections import deque


class FrameStack(gymnasium.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super(FrameStack, self).__init__(env)

        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space._shape = (k,) + env.observation_space.shape

    def reset(self):
        ob, info = self.env.reset()
        for _ in range(self.k):
            self._set_ob(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self._set_ob(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _set_ob(self, ob):
        self.frames.append(np.expand_dims(ob, axis=0))    #  add time axis

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=0)    # concat along time axis
