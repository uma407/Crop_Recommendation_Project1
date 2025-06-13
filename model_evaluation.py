import pandas as pd
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

# Load data and model
df = pd.read_csv("data/crop_recommendation.csv")
model = joblib.load("models/xgboost_model.pkl")
le = joblib.load("models/label_encoder.pkl")

X = df.drop("label", axis=1)
y = le.transform(df["label"])

y_pred = model.predict(X)

print("✅ Classification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))
