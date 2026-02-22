from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
from src.shared.load_data import download_csv
from src.shared.install_requirements import install_requirements_txt


def run(model_name="heart_disease_prediction"):
    """
    Full pipeline to train a model and start a Flask app for inference.

    Args:
        model_name (str): Name of the model folder in src/models to use for requirements.

    Workflow:
        1. Install requirements
        2. Load the data
        3. Cleanse the data
        4. Feature engineering
        5. Train the model
        6. Start the Flask app for inference
    """

    # ---------------------------
    # 1. Install requirements
    # ---------------------------
    install_requirements_txt(model_name)
    print(f"[INFO] Requirements for model '{model_name}' installed.")

    # ---------------------------
    # 2. Load the data
    # ---------------------------
    df = download_csv("heart_disease")
    print(f"[INFO] Data loaded with shape {df.shape}.")

    # ---------------------------
    # 3. Cleanse the raw data
    # ---------------------------
    df_clean = cleanse_raw_data(df)
    print(f"[INFO] RAW CSV Data cleansed; shape is now {df_clean.shape}.")

    # ---------------------------
    # 4. Feature Engineering
    # ---------------------------
    #X, y = engineer_features(df_clean)
    #print(f"[INFO] Features engineered; X shape: {X.shape}, y shape: {y.shape}.")

    # ---------------------------
    # 5. Train the model
    # ---------------------------
    #model = train_model(X, y)
    #print(f"[INFO] Model trained: {model}.")

    # ---------------------------
    # 6. Start the Flask app for inference
    # ---------------------------
    print(f"[INFO] Starting Flask app for model inference...")
    #predict_app.run(debug=True)

if __name__ == "__main__":
    run()
