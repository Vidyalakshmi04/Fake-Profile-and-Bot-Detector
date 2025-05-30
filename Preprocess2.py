import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_path = r"D:\fake-profile-detector\data\twitter_bot_detection\bot_detection_data.csv" 
df = pd.read_csv(file_path)

# Step 2: Drop unnecessary columns
df = df.drop(columns=["User ID", "Username"], errors='ignore')

# Step 3: Convert "Created At" to datetime and extract new date features
df["Created At"] = pd.to_datetime(df["Created At"], errors='coerce')
df["Account Year"] = df["Created At"].dt.year
df["Account Month"] = df["Created At"].dt.month
df["Account Day"] = df["Created At"].dt.day
df = df.drop(columns=["Created At"], errors='ignore')

# Step 4: Handle missing values in 'Hashtags'
df["Hashtags"] = df["Hashtags"].fillna("none")

# Step 5: Encode categorical features
le = LabelEncoder()
for col in ["Location", "Hashtags"]:
    df[col] = le.fit_transform(df[col].astype(str))

# Optional: Drop the Tweet column if you're not doing NLP
df = df.drop(columns=["Tweet"], errors='ignore')

# Step 6: Save the cleaned dataset
output_path = r"D:\fake-profile-detector\cleaned_twitter_bot_data1.csv"  
df.to_csv(output_path, index=False)

print("✅ Preprocessing complete. File saved to:", output_path)
