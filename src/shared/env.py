import os
from keras.metrics import AUC


# PROJECT_ROOT
# - Local dev: nn-playground folder
# - Docker: /app
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

# Paths
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GLOBAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # <-- global models folder
HEART_DISEASE_MODEL_NAME = "heart_disease_prediction"

# Default CSVs
HEART_DISEASE_CSV = os.path.join(DATA_RAW_DIR, "heart_disease.csv")

# URLs for downloading data
CSV_RAW_DATASET_URLS = {
    "heart_disease": "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
}

# Columns that are numeric fields
NUMERIC_COLUMNS = ['age', 'trestbps','chol', 'thalach', 'oldpeak', 'slope']

# Cleaning thresholds
MISSING_VALUE_THRESHOLD = 70  # % threshold for dropping columns

# Values to drop for specific categorical columns
INVALID_CATEGORIES = {
    "thal": ["1", "2"]
}

# Model hyperparameters (example, can expand per model)
MODEL_PARAMS = {
    "heart_disease_prediction": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "hidden_layers": [16],
        "activation": "relu",
        "output_activation": "sigmoid",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", AUC(name="auc")],
        "epochs": 50,
        "batch_size": 16,
        "validation_split": 0.2
    }
}