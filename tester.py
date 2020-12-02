# main libraries
import gym
import numpy as np
import wandb
import argparse

# register PyBullet enviroments with OpenAI Gym
import pybulletgym

from tensorflow import keras

# Herne prostredie
env = gym.make('Walker2DPyBulletEnv-v0')
env.render()

# load actor model
model = keras.models.load_model("save/model_A.h5")


# main loop
for t in range(5):
    done = False
    episode_reward, episode_timesteps = 0.0, 0

    obs = env.reset()

    while not done:
        #env.render()

        mu, sigma = model(np.expand_dims(obs, axis=0))
        print(f'mu: {mu}')
        print(f'sigma: {sigma}\n')

        new_obs, reward, done, _ = env.step(mu[0])

        episode_reward += reward
        episode_timesteps += 1

        # critical !!!
        obs = new_obs

    print(f'Epoch: {t}')
    print(f'Score: {episode_reward}')
    print(f'Steps: {episode_timesteps}\n')

env.close()