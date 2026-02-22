from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
from src.models.heart_disease_prediction.engineer_features import engineer_features
from src.shared.load_data import download_csv
from src.shared.install_requirements import install_requirements_txt

from colorist import green


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
    green(f"************+ START: cleanse_raw_data ************+")
    df_clean = cleanse_raw_data(df)
    green(f"[INFO] RAW CSV Data cleansed; shape is now {df_clean.shape}.")
    green(f"************+ END: cleanse_raw_data ************+")

    # ---------------------------
    # 4. Feature Engineering
    # ---------------------------
    green(f"************+ START: engineer_features ************+")
    X, y = engineer_features(df_clean)
    green(f"[INFO] Features engineered; X shape: {X.shape}, y shape: {y.shape}.")
    green(f"************+ END: engineer_features ************+")

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
