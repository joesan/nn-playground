from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
from src.models.heart_disease_prediction.engineer_features import *
from src.shared.load_data import download_csv
from src.shared.install_requirements import install_requirements_txt
from sklearn.model_selection import train_test_split

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
    # 4. Feature Engineering - Before Test & Train data split
    # ---------------------------
    green(f"************+ START: engineer_features ************+")
    # Encode categorical columns
    df_encoded = encode_categoricals(df_clean)

    # ---------------------------
    # 5. Split the encoded dataset
    # ---------------------------
    print("Splitting test and train data using test_size=0.2, random_state=42")
    train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_clean["target"])

    # ---------------------------
    # 6. Feature Engineering - After Test & Train data split
    # ---------------------------
    # Split feature and target for train dataset
    X_train, y_train = split_target(train_df, target_column="target")
    X_test, y_test = split_target(test_df, target_column="target")

    # Scale the numeric columns in the training dataset
    X_train_scaled, y_train, scaler = scale_numeric_features(X_train, y_train)

    # Apply the scaler to the test dataset
    X_test_scaled, y_test, _ = scale_numeric_features(X_test, y_test)
    
    green(f"[INFO] Features engineered; X_train shape: {X_train_scaled.shape}, y shape: {y_train.shape}.")
    green(f"[INFO] Features engineered; X_test shape:  {X_test_scaled.shape},  y shape: {y_test.shape}.")
    green(f"************+ END: engineer_features ************+")

    # ---------------------------
    # 7. Train the model
    # ---------------------------
    #model = train_model(X, y)
    #print(f"[INFO] Model trained: {model}.")

    # ---------------------------
    # 8. Start the Flask app for inference
    # ---------------------------
    print(f"[INFO] Starting Flask app for model inference...")
    #predict_app.run(debug=True)

if __name__ == "__main__":
    run()
