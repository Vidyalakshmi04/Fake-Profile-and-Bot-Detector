import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# === CONFIGURATION ===
# === CONFIGURATION ===
MODEL_PATH = r"D:\fake-profile-detector\data\face_detection\fake_image_detector_model.h5"
TEST_DIR = r"D:\fake-profile-detector\data\face_detection\test_images"
IMAGE_SIZE = (128, 128)                # Same as used in training
CLASS_NAMES = ['real', 'fake']         # Based on training

# === Load Model ===
model = load_model(MODEL_PATH)

# === Load and Preprocess Images ===
def load_test_images(test_dir):
    X = []
    y = []
    for label in CLASS_NAMES:
        folder = os.path.join(test_dir, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder, fname)
                img = image.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize
                X.append(img_array)
                y.append(CLASS_NAMES.index(label))
    return np.array(X), np.array(y)

X_test, y_test = load_test_images(TEST_DIR)

# === Predict ===
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# === Evaluation ===
print("Fake Profile Image Detection Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
