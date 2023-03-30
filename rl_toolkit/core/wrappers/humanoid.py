import time

import gymnasium
import numpy as np
from dm_control.utils import rewards
from gymnasium import spaces


class HumanoidRobot(gymnasium.Env):
    metadata = {"tasks": ["stand", "walk"], "render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, port="/dev/ttyACM0"):
        self.render_mode = render_mode
        self.task = "stand"
        self.port = port

        # action info
        action_shape = (6,)
        self.action_space = spaces.Box(
            np.full(action_shape, -1.0, dtype=np.float32),
            np.full(action_shape, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # observation info
        observation_shape = (16,)
        self.observation_space = spaces.Box(
            np.full(observation_shape, -np.inf, dtype=np.float32),
            np.full(observation_shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def connect(self):
        import serial

        # init serial port
        self.comm = serial.Serial(self.port, baudrate=115200)
        time.sleep(3)

        # read booting stage
        while self.comm.in_waiting:
            print(self.comm.readline())

    def _get_obs(self):
        # read observation from arduino
        self.comm.write(b"S")
        data = self.comm.readline()
        obs = np.fromstring(data, dtype=np.float32, sep=";")
        return obs

    def _get_reward(self, obs):
        if self.task == "stand":
            roll, pitch = obs[12], obs[13]
            # Euclidean distance
            mag = np.sqrt(np.square(roll) + np.square(pitch))
            print(mag)

            return rewards.tolerance(
                mag,
                bounds=(0.0, 5.0),  # ±5 degrees
                sigmoid="linear",
                margin=90.0,
                value_at_margin=0.0,
            )

    def _set_action(self, action):
        # send action to arduino
        m_bytes = "M"
        for i in range(action.shape[0]):
            m_bytes += np.format_float_positional(action[i]) + ";"
        self.comm.write(m_bytes.encode("ascii"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # send action to arduino
        self._set_action(action)

        # delay for servo stabilization
        time.sleep(0.1)

        # read observation from arduino
        obs = self._get_obs()

        print(obs)

        return (
            obs,
            self._get_reward(obs),
            False,
            False,
            {},
        )

    def close(self):
        self.comm.close()
