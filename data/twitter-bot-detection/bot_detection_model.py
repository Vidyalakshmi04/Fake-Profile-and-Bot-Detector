# train_bot_detection_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv(r"D:\fake-profile-detector\data\twitter_bot_detection\bot_detection_data.csv")

# Convert 'Created At' to datetime and extract day/month/year
data['Created At'] = pd.to_datetime(data['Created At'], errors='coerce')
data['Account Day'] = data['Created At'].dt.day
data['Account Month'] = data['Created At'].dt.month
data['Account Year'] = data['Created At'].dt.year

# Drop columns you don't want to use as features
columns_to_drop = ['User ID', 'Username', 'Tweet', 'Location', 'Created At', 'Hashtags', 'Bot Label']
features = data.drop(columns=columns_to_drop)

# Target variable
labels = data['Bot Label']

# Handle categorical columns if any (e.g., 'Verified' if it's a string)
if features['Verified'].dtype == 'object':
    features['Verified'] = features['Verified'].map({'True': 1, 'False': 0})

# Fill or drop any NaNs
features = features.fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, r"D:\fake-profile-detector\data\twitter_bot_detection\profile_bot_rf_model.pkl")
joblib.dump(scaler, r"D:\fake-profile-detector\data\twitter_bot_detection\profile_bot_scaler.pkl")
