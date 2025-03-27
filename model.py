### Step 3: Load and predict with AI model
# model.py
import tensorflow as tf
import numpy as np

def load_action_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        print(f"Model {model_path} not found.")
        exit()

def predict_action_sequence(model, keypoint_sequence):
    keypoint_sequence = np.expand_dims(keypoint_sequence, axis=0)
    predictions = model.predict(keypoint_sequence)[0]
    actions = ['Walking', 'Running', 'Jumping', 'Waving', 'Sitting', 'Falling', 'Dancing', 'Fighting', 'Unknown']
    confidence = max(predictions)
    return actions[np.argmax(predictions)], confidence
