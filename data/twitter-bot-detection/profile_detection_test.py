import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load the trained model and scaler
model = joblib.load("D:/fake-profile-detector/data/fake_profile_detection/profile_detection_model.pkl")
scaler = joblib.load("D:/fake-profile-detector/data/fake_profile_detection/profile_detection_scaler.pkl")

# Load test data
test_df = pd.read_csv("D:\\fake-profile-detector\\data\\fake_profile_detection\\cleaned_fake_profiles1.csv")

# Drop non-numeric and unnecessary columns
test_df = test_df.drop(columns=["lang", "location", "description"], errors='ignore')

# Convert boolean columns to int
test_df["default_profile"] = test_df["default_profile"].astype(int)
test_df["profile_use_background_image"] = test_df["profile_use_background_image"].astype(int)

# Separate features and label
X_test = test_df.drop("dataset", axis=1)
y_test = test_df["dataset"]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
print("Fake Profile Detection Model Evaluation:")
print(classification_report(y_test, y_pred))
