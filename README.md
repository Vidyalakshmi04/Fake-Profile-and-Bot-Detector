# 🕵️‍♂️ AI-Based Fake Profile & Bot Detection System

A unified AI system for detecting fake profiles and bots on social media (e.g., Twitter). It uses machine learning and deep learning models to analyze both **user metadata** and **profile images**. Deployed as a **Flask web app**, it provides real-time predictions through a clean user interface.



##  Overview

The system integrates three detection modules:

-  **Bot Detection:** Uses metadata features with a Random Forest Classifier.
-  **Fake Profile Detection:** ML-based classifier trained on real/fake account data.
-  **Fake Image Detection:** CNN model trained to detect AI-generated profile images.
   
    <p><img src="https://github.com/user-attachments/assets/b7c91632-a2ba-46b3-b270-b621f0a2a716" width="350"/></p>
    <p><img src="https://github.com/user-attachments/assets/b143d48f-7c95-43b8-8363-ba222e8f84c6" width="350"/></p>

##  Features

- Multi-model pipeline for robust detection  
- CNN-based profile image analysis  
- Real-time metadata evaluation  
- Color-coded prediction outputs with probabilities  
- Flask API with a responsive web UI



##  Tech Stack

- Python, Flask  
- HTML, CSS, JavaScript  
- TensorFlow/Keras (CNN)  
- Scikit-learn (Random Forest)  
- OpenCV / PIL (Image preprocessing)



##  Thresholds

- **Image Classifier:**  
  - `< 0.5` → ✅ Real  
  - `≥ 0.5` → ❌ Fake



##  License

Licensed under the **MIT License** – feel free to use, modify, and share with attribution.
