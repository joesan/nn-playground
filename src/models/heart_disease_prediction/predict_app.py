from flask import Flask, request, jsonify
import os, subprocess, sys
import joblib
from tensorflow import keras
import pandas as pd
import src.shared.env as env


# ---------------------------
# 1. Initialize Flask
# ---------------------------
app = Flask(__name__)


# ---------------------------
# 2. Load model, scaler and column names
# ---------------------------
MODEL_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}.keras")
model = keras.models.load_model(MODEL_PATH)

SCALER_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}_scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# Load training columns (after one-hot encoding)
COLUMNS_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}_columns.pkl")
columns = joblib.load(COLUMNS_PATH)


# ---------------------------
# 3. Define endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload with features and returns model prediction.
    """
    data = request.get_json()
    df = pd.DataFrame([data])

    # -------------------------
    # 1. Encode categorical columns (one-hot)
    # -------------------------
    categorical_cols = [c for c in df.columns if c not in env.NUMERIC_COLUMNS]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # -------------------------
    # 2. Align columns with training data
    # -------------------------
    for col in columns:
        if col not in df.columns:
            df[col] = 0  # add missing columns with 0
    df = df[columns]  # reorder columns

    # -------------------------
    # 3. Scale numeric columns
    # -------------------------
    df[env.NUMERIC_COLUMNS] = scaler.transform(df[env.NUMERIC_COLUMNS])

    # -------------------------
    # 4. Make prediction
    # -------------------------
    pred_prob = model.predict(df)[0][0]
    pred_class = int(pred_prob >= 0.5)

    return jsonify({
        "prediction": pred_class,
        "probability": float(pred_prob)
    })


# ---------------------------
# 4. Run app
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Starting Flask app for model inference...")
    app.run(debug=True, host="0.0.0.0", port=5000)