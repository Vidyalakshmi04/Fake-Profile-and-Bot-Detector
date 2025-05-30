# 🕵️‍♂️ AI-Based Fake Profile & Bot Detection System

A unified AI system for detecting fake profiles and bots on social media (e.g., Twitter). It uses machine learning and deep learning models to analyze both **user metadata** and **profile images**. Deployed as a **Flask web app**, it provides real-time predictions through a clean user interface.



##  Overview

The system integrates three detection modules:

-  **Bot Detection:** Uses metadata features with a Random Forest Classifier.
-  **Fake Profile Detection:** ML-based classifier trained on real/fake account data.
-  **Fake Image Detection:** CNN model trained to detect AI-generated profile images.
   
    <p><img src="https://github.com/user-attachments/assets/4de01026-7f4d-4be7-87f9-0069cc565ec8" width="400"/></p>
    <p><img src="https://github.com/user-attachments/assets/45466e68-dc6c-4f74-99ba-c1e51e1ba312" width="400"/></p>

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