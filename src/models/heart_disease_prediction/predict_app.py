from flask import Flask, request, jsonify
import os, subprocess, sys
import joblib
from tensorflow import keras
import pandas as pd
import src.shared.env as env


# ---------------------------
# Step 0: Ensure model-specific venv exists
# ---------------------------
def create_model_venv(model_name: str, requirements_path: str = None):
    """
    Create a model-specific virtual environment inside the main .venv folder.
    """
    # Determine project root dynamically
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # Ensure main .venv folder exists
    main_venv_dir = os.path.join(PROJECT_ROOT, ".venv")
    os.makedirs(main_venv_dir, exist_ok=True)

    # Model-specific venv path inside main .venv
    venv_path = os.path.join(main_venv_dir, f"{model_name}")

    if not os.path.exists(venv_path):
        print(f"[INFO] Creating virtual environment for model '{model_name}' at {venv_path} ...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    else:
        print(f"[INFO] Virtual environment already exists at {venv_path} ...")

    # Determine pip executable in the venv
    pip_exec = os.path.join(venv_path, "Scripts", "pip.exe") if os.name == "nt" else os.path.join(venv_path, "bin",
                                                                                                  "pip")

    # Use provided requirements.txt or fallback to global one
    if requirements_path is None:
        requirements_path = os.path.join(PROJECT_ROOT, "requirements.txt")

    print(f"[INFO] Installing dependencies from {requirements_path} ...")
    subprocess.check_call([pip_exec, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_exec, "install", "-r", requirements_path])

    print(f"[INFO] Virtual environment for model '{model_name}' is ready at {venv_path}")
    return venv_path

# ---------------------------
# Step 1: Create venv for the model
# ---------------------------
create_model_venv(env.HEART_DISEASE_MODEL_NAME)

# ---------------------------
# 2. Initialize Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# 3. Load model, scaler and column names
# ---------------------------
MODEL_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}.keras")
model = keras.models.load_model(MODEL_PATH)

SCALER_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}_scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# Load training columns (after one-hot encoding)
COLUMNS_PATH = os.path.join(env.GLOBAL_MODELS_DIR, f"{env.HEART_DISEASE_MODEL_NAME}_columns.pkl")
columns = joblib.load(COLUMNS_PATH)


# ---------------------------
# 4. Define endpoint
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
# 5. Run app
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Starting Flask app for model inference...")
    app.run(debug=True, host="0.0.0.0", port=5000)