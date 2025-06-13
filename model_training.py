# Model training script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/crop_recommendation.csv")

# Split features and label
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model and label encoder
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Model and encoder saved in /models/")
