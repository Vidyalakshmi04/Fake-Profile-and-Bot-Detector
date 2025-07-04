import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Paths
real_dir = r"D:\fake-profile-detector\data\face_detection\processedtraining_real"
fake_dir = r"D:\fake-profile-detector\data\face_detection\processedtraining_fake"

IMG_SIZE = 128

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

X = []
y = []

# Load real images
for filename in os.listdir(real_dir):
    path = os.path.join(real_dir, filename)
    img = preprocess_image(path)
    if img is not None:
        X.append(img)
        y.append(0)  # Label: Real

# Load fake images
for filename in os.listdir(fake_dir):
    path = os.path.join(fake_dir, filename)
    img = preprocess_image(path)
    if img is not None:
        X.append(img)
        y.append(1)  # Label: Fake

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("fake_image_detector_model.h5")
