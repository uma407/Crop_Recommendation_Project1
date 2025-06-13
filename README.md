# Crop Recommendation System

This project predicts the most suitable crop to grow based on environmental conditions using a machine learning model trained on agricultural data.

## 📁 Folder Structure

```
crop-recommendation-system/
├── app.py                  # Flask API backend
├── train_model.py          # XGBoost training script
├── crop_model.joblib       # Trained model (after running train_model.py)
├── label_encoder.joblib    # Label encoder (after training)
├── crop_dataset.csv        # Dataset (you must place this here)
├── index.html              # Frontend UI for input and crop output
└── README.md               # Project description and usage
```

## 🚀 How to Run This Project

### Step 1: Install dependencies
```bash
pip install flask xgboost scikit-learn pandas joblib flask-cors
```

### Step 2: Train the model
```bash
python train_model.py
```

This generates `crop_model.joblib` and `label_encoder.joblib`.

### Step 3: Run the Flask API
```bash
python app.py
```

The server will start at: `http://127.0.0.1:5000`

### Step 4: Open the Frontend
Simply open `index.html` in a browser. Submit values and see the predicted crop!

## 🧠 Model Info
- **Algorithm:** XGBoost Classifier
- **Input Features:** Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- **Target:** Recommended Crop (22 categories)

## 📊 Dataset
- Total samples: ~2200
- Format: CSV
- Required column headers:
  `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`

## 🛠 Deployment
Optional: You can deploy this using platforms like **Render**, **Railway**, or **Replit**.

## 👩‍💻 Author
UMAMAHESWARI BOPPANA

## ✅ Result
Tested with local API and UI — working end-to-end crop recommendation system based on user inputs.
