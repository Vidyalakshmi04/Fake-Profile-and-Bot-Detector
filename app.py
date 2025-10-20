# streamlit_app_account_image_gradcam_fixed.py
import streamlit as st
import joblib
from tensorflow.keras.models import load_model, Model
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

# --- Load models ---
bot_model = joblib.load(r"D:\fake-profile-detector\data\twitter_bot_detection\profile_bot_rf_model.pkl")
profile_model = joblib.load(r"D:\fake-profile-detector\data\fake_profile_detection\profile_detection_model.pkl")
image_model = load_model(r"D:\fake-profile-detector\data\face_detection\fake_image_detector_model.h5")

# --- Feature order ---
BOT_FEATURE_ORDER = [
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "default_profile",
    "profile_use_background_image"
]

PROFILE_FEATURE_ORDER = [
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "default_profile",
    "profile_use_background_image", "has_description",
    "has_location", "verified", "lang_encoded",
    "location_encoded", "description_length"
]

# --- Prediction functions ---
def predict_bot(features: dict):
    X = np.array([[features[f] for f in BOT_FEATURE_ORDER]])
    return float(bot_model.predict_proba(X)[0][1])

def predict_profile(features: dict):
    X = np.array([[features[f] for f in PROFILE_FEATURE_ORDER]])
    return float(profile_model.predict_proba(X)[0][1])

def predict_image(img: Image.Image):
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)/255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return float(image_model.predict(img_array)[0][0])

# --- Grad-CAM function ---
def generate_gradcam(img: Image.Image, model):
    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)/255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        return None

    # Handle multiple outputs safely
    model_output = model.output
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0]

    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model_output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # Overlay heatmap
    heatmap = cv2.resize(heatmap.numpy(), (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

# --- Streamlit App ---
st.set_page_config(page_title="Fake Profile & Image Detector", layout="wide")
st.markdown("""
<style>
.stApp {background-color: rgba(0,0,0,0.85); color: #e0e0e0;}
.stButton>button {background-color: #7C3AED; color: white; font-size: 16px; padding: 10px 20px;}
.result-box {padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; margin-bottom: 15px; color: #ffffff; font-size: 18px;}
.real {background-color: #2E7D32;}
.suspicious {background-color: #FBC02D;}
.fake {background-color: #C62828;}
.table-box {background-color:#424242; color:white; padding:10px; border-radius:10px; margin-top:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#BB86FC;'>AI-Based Fake Profile & Image Detector</h1>", unsafe_allow_html=True)

# Upload image
st.subheader("Upload Profile Image")
uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

# Input features
with st.expander("Bot & Profile Features", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        statuses_count = st.number_input("Statuses Count", min_value=0, value=500)
        followers_count = st.number_input("Followers Count", min_value=0, value=300)
        friends_count = st.number_input("Friends Count", min_value=0, value=400)
    with c2:
        favourites_count = st.number_input("Favourites Count", min_value=0, value=50)
        listed_count = st.number_input("Listed Count", min_value=0, value=10)
        default_profile = st.selectbox("Default Profile (0 or 1)", [1, 0])
    with c3:
        profile_use_background_image = st.selectbox("Profile Background Image (0 or 1)", [1, 0])

with st.expander("Profile Additional Features", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        has_description = st.selectbox("Has Description (0 or 1)", [1, 0])
        has_location = st.selectbox("Has Location (0 or 1)", [1, 0])
    with c2:
        verified = st.selectbox("Verified (0 or 1)", [0, 1])
        lang_encoded = st.number_input("Language Encoded (int)", value=0)
    with c3:
        location_encoded = st.number_input("Location Encoded (int)", value=0)
        description_length = st.number_input("Description Length", min_value=0, value=0)

# Detect button
if st.button("Detect"):
    # --- Account Classification ---
    bot_features = {
        "statuses_count": statuses_count, "followers_count": followers_count,
        "friends_count": friends_count, "favourites_count": favourites_count,
        "listed_count": listed_count, "default_profile": default_profile,
        "profile_use_background_image": profile_use_background_image
    }
    profile_features = {
        **bot_features, "has_description": has_description,
        "has_location": has_location, "verified": verified,
        "lang_encoded": lang_encoded, "location_encoded": location_encoded,
        "description_length": description_length
    }

    bot_prob = predict_bot(bot_features)
    profile_prob = predict_profile(profile_features)
    account_prob = 0.4*bot_prob + 0.6*profile_prob

    # Account label
    if account_prob < 0.4:
        account_label = "🟢 REAL ACCOUNT"
        account_class = "real"
        account_reason = "Bot behavior and profile metadata look normal."
    elif 0.4 <= account_prob < 0.7:
        account_label = "🟡 SUSPICIOUS ACCOUNT"
        account_class = "suspicious"
        account_reason = "Some anomalies in metadata or bot-like behavior detected."
    else:
        account_label = "🔴 FAKE ACCOUNT"
        account_class = "fake"
        account_reason = "Profile and bot metrics indicate fake account."

    st.markdown(f"<div class='result-box {account_class}'>{account_label}<br>Probability: {account_prob*100:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='table-box'><b>Why:</b> {account_reason}</div>", unsafe_allow_html=True)
    st.subheader("📊 Account Model Probabilities")
    st.markdown(f"- 🤖 Bot Probability: {bot_prob*100:.2f}%")
    st.markdown(f"- 🧩 Profile Probability: {profile_prob*100:.2f}%")

    # --- Image Classification ---
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_prob = predict_image(image)
        if image_prob < 0.5:
            image_label = "🟢 REAL IMAGE"
            image_class = "real"
            image_reason = "Image looks natural and unaltered."
        else:
            image_label = "🔴 FAKE IMAGE"
            image_class = "fake"
            image_reason = "CNN detects AI-generated or manipulated patterns."

        st.markdown(f"<div class='result-box {image_class}'>{image_label}<br>Probability: {image_prob*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='table-box'><b>Why:</b> {image_reason}</div>", unsafe_allow_html=True)

# Detection modules explanation
st.subheader("Detection Modules:")
st.markdown("Bot Detection: Random Forest Classifier (follower patterns, tweet activity, etc.)")
st.markdown("Fake Profile Detection: ML model trained on real vs. fake account data")
st.markdown("Fake Image Detection: CNN model to detect AI-generated profile photos")
