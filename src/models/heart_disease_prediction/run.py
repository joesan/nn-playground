from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
from src.models.heart_disease_prediction.engineer_features import *
from src.shared.load_data import download_csv
from src.shared.install_requirements import install_requirements_txt
from src.models.heart_disease_prediction.build_model import *
from sklearn.model_selection import train_test_split
import numpy as np
from colorist import green
import joblib


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
    X_train_split, y_train_split = split_target(train_df, target_column="target")
    X_test_split, y_test_split = split_target(test_df, target_column="target")

    # Scale the numeric columns in the training dataset
    X_train, y_train, scaler = scale_numeric_features(X_train_split, y_train_split)
    # Apply the scaler to the test dataset
    X_test, y_test, _ = scale_numeric_features(X_test_split, y_test_split, scaler)

    # Save the scaler
    scaler_path = os.path.join(env.GLOBAL_MODELS_DIR, f"{model_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    # Save the columns
    columns_path = os.path.join(env.GLOBAL_MODELS_DIR, f"{model_name}_columns.pkl")
    joblib.dump(X_train.columns.tolist(), columns_path)

    green(f"[INFO] Features engineered; X_train shape: {X_train.shape},  y shape: {y_train.shape}.")
    green(f"[INFO] Features engineered; X_train dtype: {X_train.dtypes}, y dtype: {y_train.dtypes}.")
    green(f"[INFO] Features engineered; X_test shape:  {X_test.shape},   y dtype: {y_test.shape}.")
    green(f"[INFO] Features engineered; X_test dtype:  {X_test.dtypes},  y dtype: {y_test.dtypes}.")
    green(f"************+ END: engineer_features ************+")

    # ---------------------------
    # 7. Build the model
    # ---------------------------
    X_train = X_train.astype(np.float32).values
    y_train = y_train.astype(np.float32).values

    X_test = X_test.astype(np.float32).values
    y_test = y_test.astype(np.float32).values

    green(f"************+ START: build_model ************+")
    model = build_model(input_dim=X_train.shape[1], model_name="heart_disease_prediction")
    green(f"************+ END: build_model ************+")

    # ---------------------------
    # 8. Train the model
    # ---------------------------
    model, history = train_model(
        model,
        X_train, y_train,
        model_name="heart_disease_prediction"
    )

    # ---------------------------
    # 8. Measure the accuracy
    # ---------------------------
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    print(results)
    # Example output:
    # {'loss': 0.3576, 'accuracy': 0.8197, 'auc': 0.89}

    loss = results['loss']
    accuracy = results['accuracy']
    auc = results['auc']

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    run()
