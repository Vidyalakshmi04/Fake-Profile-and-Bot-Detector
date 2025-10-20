# 🧠 AI-Based Fake Profile & Image Detector

An intelligent web application that detects **fake, suspicious, or real social media accounts** and verifies whether a **profile image is real or AI-generated**, using **Machine Learning + Deep Learning** .



## 🚀 Features

✅ **Dual Detection System:**  
- **Account Classification:** Real / Suspicious / Fake based on bot-like activity and profile metadata.  
- **Image Classification:** Real / Fake detection using a CNN trained on authentic vs. AI-generated faces.  

✅ **Interactive Streamlit Interface:**  
- Upload a profile image and enter account metrics through a clean, animated UI.  
- Visual and textual outputs that explain every result.  

✅ **Weighted Hybrid Model:**  
- Combines predictions from a **bot classifier** and **fake profile classifier** for more accurate account labeling.  



## 🧩 Architecture Overview

| Component | Model Type | Framework | Purpose |
|------------|-------------|-----------|----------|
| **Bot Detection** | RandomForestClassifier | Scikit-Learn | Identifies automated (bot-like) accounts |
| **Profile Classification** | RandomForestClassifier | Scikit-Learn | Detects fake/suspicious accounts using profile metadata |
| **Image Detection** | Convolutional Neural Network | TensorFlow/Keras | Classifies uploaded image as Real or Fake |



## 🧠 Model Inputs

### 🔹 Bot & Profile Features
| Feature | Description |
|----------|-------------|
| statuses_count | Total tweets/posts by user |
| followers_count | Number of followers |
| friends_count | Number of accounts followed |
| favourites_count | Likes given |
| listed_count | Times user is added to lists |
| default_profile | Uses default layout (1/0) |
| profile_use_background_image | Uses default background (1/0) |
| has_description | Whether description is provided |
| has_location | Whether location is visible |
| verified | Account verified badge (1/0) |
| lang_encoded | Encoded language of user |
| location_encoded | Encoded location data |
| description_length | Length of bio description |

### 🔹 Image Input
- Accepts `.png`, `.jpg`, or `.jpeg`  
- Resized to **128×128**  
- CNN predicts **Real** or **Fake**



## 🧮 Working Logic

1. **Bot Classifier** → predicts probability of automation.  
2. **Profile Classifier** → predicts likelihood of fake/suspicious behavior.  
3. Weighted score combines both → final account label:  
       < 0.4 → 🟢 Real Account  
       0.4–0.7 → 🟡 Suspicious Account  
       > 0.7 → 🔴 Fake Account  
4. CNN predicts image authenticity.


