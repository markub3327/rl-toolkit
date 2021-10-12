import tensorflow as tf
import numpy as np
import gym
import pybullet_envs  # noqa

from rl_toolkit.networks.models import Actor

# Init environment
env = gym.make('MinitaurBulletEnv-v0')

# Init model
model = Actor(n_outputs=np.prod(env.action_space.shape))


# Run on random data
model(tf.random.normal((1,) + env.action_space.shape))

print('Model initialized 👍')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print('Model converted 👍')

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print('TFLite model saved 🔥')

env.close()