import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# -------------------------------
# Load models
bot_model = joblib.load("D:\\fake-profile-detector\\data\\twitter_bot_detection\\profile_bot_rf_model.pkl")
profile_model = joblib.load("D:\\fake-profile-detector\\data\\fake_profile_detection\\profile_detection_model.pkl")
image_model = load_model("D:\\fake-profile-detector\\data\\face_detection\\fake_image_detector_model.h5")

# -------------------------------
# Features for bot_model (7 total)
bot_features = [
    500,  # statuses_count
    300,  # followers_count
    400,  # friends_count
    50,   # favourites_count
    10,   # listed_count
    1,    # default_profile
    1     # profile_use_background_image
]

# Features for profile_model (13 total — must match model training structure exactly)
profile_features = [
    500,   # statuses_count
    300,   # followers_count
    400,   # friends_count
    50,    # favourites_count
    10,    # listed_count
    1,     # default_profile
    1,     # profile_use_background_image
    1,     # has_description
    1,     # has_location
    0,     # verified
    0,     # lang_encoded
    0,     # location_encoded
    0      # description_length or other numerical feature
]

# -------------------------------
# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (128, 128))    # Match training input
    image = image / 255.0                    # Normalize
    image = np.expand_dims(image, axis=0)    # Add batch dimension: (1, 128, 128, 3)
    return image


image_input = preprocess_image("D:\\fake-profile-detector\\data\\face_detection\\test_images\\real\\real_00051.jpg")
image_prob = image_model.predict(image_input)[0][0]


# -------------------------------
# Predict
bot_prob = bot_model.predict_proba([bot_features])[0][1]
profile_prob = profile_model.predict_proba([profile_features])[0][1]
image_prob = image_model.predict(image_input)[0][0]

# -------------------------------
# Output
print("Bot Probability (0 = human, 1 = bot):", bot_prob)
print("Fake Profile Probability (0 = real, 1 = fake):", profile_prob)
print("Fake Image Probability (0 = real, 1 = fake):", image_prob)
