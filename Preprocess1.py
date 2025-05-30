import pandas as pd
import numpy as np
# Load the datasets
df1 = pd.read_csv("D:\\fake-profile-detector\\data\\fake_profile_detection\\fusers.csv")
df2 = pd.read_csv("D:\\fake-profile-detector\\data\\fake_profile_detection\\users.csv")
# Combine the datasets
df = pd.concat([df1, df2], ignore_index=True)
# Drop columns with more than 50% missing values
df = df.dropna(thresh=len(df) * 0.5, axis=1)
columns_to_drop = [
    'id', 'name', 'screen_name', 'created_at', 'updated', 'profile_image_url',
    'profile_image_url_https', 'profile_banner_url', 'profile_background_image_url',
    'profile_background_image_url_https', 'url'
]
df = df.drop(columns=columns_to_drop, errors='ignore')
df = df.fillna({
    col: 0 for col in df.select_dtypes(include=[np.number]).columns
})
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in df.select_dtypes(include=[object]).columns:
    df[column] = le.fit_transform(df[column])
df.to_csv("cleaned_fake_profiles.csv", index=False)
#import os
#print(os.getcwd())
