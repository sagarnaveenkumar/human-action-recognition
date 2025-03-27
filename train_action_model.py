import tensorflow as tf
model = tf.keras.models.load_model('action_recognition_model.h5')
print(model.summary())
