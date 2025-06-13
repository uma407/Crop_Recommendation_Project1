import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/xgboost_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

st.title("🌾 Crop Recommendation System")
st.write("Enter the agricultural parameters below:")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    pred_probs = model.predict_proba(input_data)[0]
    top_3_idx = pred_probs.argsort()[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_idx)
    st.success(f"Top 3 Recommended Crops:")
    for crop, score in zip(top_3_crops, pred_probs[top_3_idx]):
        st.write(f"🌱 {crop} — Confidence: {score:.2%}")
