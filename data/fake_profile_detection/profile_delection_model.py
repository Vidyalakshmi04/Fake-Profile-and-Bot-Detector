import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("D:\\fake-profile-detector\\data\\fake_profile_detection\\cleaned_fake_profiles1.csv")

# Drop columns that are non-numeric and less useful directly
df = df.drop(columns=["lang", "location", "description"], errors='ignore')

# Convert boolean columns to numeric
df["default_profile"] = df["default_profile"].astype(int)
df["profile_use_background_image"] = df["profile_use_background_image"].astype(int)

# Separate features and label
X = df.drop("dataset", axis=1)
y = df["dataset"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Fake Profile Detection Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "D:/fake-profile-detector/data/fake_profile_detection/profile_detection_model.pkl")
joblib.dump(scaler, "D:/fake-profile-detector/data/fake_profile_detection/profile_detection_scaler.pkl")
