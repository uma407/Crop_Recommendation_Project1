from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/xgboost_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    probs = model.predict_proba(input_df)[0]
    top_3_idx = probs.argsort()[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_idx)
    return jsonify({
        "recommendations": [
            {"crop": top_3_crops[i], "confidence": float(probs[top_3_idx[i]])}
            for i in range(3)
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
# Flask API code here