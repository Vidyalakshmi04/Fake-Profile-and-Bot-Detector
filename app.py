from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load models
bot_model = joblib.load("D:\\fake-profile-detector\\data\\twitter_bot_detection\\profile_bot_rf_model.pkl")
profile_model = joblib.load("D:\\fake-profile-detector\\data\\fake_profile_detection\\profile_detection_model.pkl")
image_model = load_model("D:\\fake-profile-detector\\data\\face_detection\\fake_image_detector_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from form
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_path = "temp.jpg"
    image_file.save(image_path)

    # Preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Extract bot features from form inputs
    try:
        bot_features = [
            int(request.form['statuses_count']),
            int(request.form['followers_count']),
            int(request.form['friends_count']),
            int(request.form['favourites_count']),
            int(request.form['listed_count']),
            int(request.form['default_profile']),
            int(request.form['profile_use_background_image'])
        ]

        profile_features = [
            int(request.form['statuses_count']),
            int(request.form['followers_count']),
            int(request.form['friends_count']),
            int(request.form['favourites_count']),
            int(request.form['listed_count']),
            int(request.form['default_profile']),
            int(request.form['profile_use_background_image']),
            int(request.form['has_description']),
            int(request.form['has_location']),
            int(request.form['verified']),
            int(request.form['lang_encoded']),
            int(request.form['location_encoded']),
            int(request.form['description_length'])
        ]
    except Exception as e:
        return jsonify({'error': f'Invalid or missing form data: {e}'}), 400

    # Make predictions
    bot_prob = float(bot_model.predict_proba([bot_features])[0][1])
    profile_prob = float(profile_model.predict_proba([profile_features])[0][1])
    image_prob = float(image_model.predict(image)[0][0])

    return jsonify({
        'bot_prob': bot_prob,
        'profile_prob': profile_prob,
        'image_prob': image_prob
    })


if __name__ == '__main__':
    app.run(debug=True)
