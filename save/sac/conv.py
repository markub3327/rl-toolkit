import tensorflow as tf


model = tf.keras.models.load_model("/home/martin/Projects/rl-baselines/save/sac/model_A_BipedalWalker.h5")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
