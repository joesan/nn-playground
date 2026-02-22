import os

# Determine project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Paths
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GLOBAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # <-- global models folder

# Default CSVs
HEART_DISEASE_CSV = os.path.join(DATA_RAW_DIR, "heart_disease.csv")

# URLs for downloading data
CSV_RAW_DATASET_URLS = {
    "heart_disease": "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
}

# Cleaning thresholds
MISSING_VALUE_THRESHOLD = 70  # % threshold for dropping columns

# Model hyperparameters (example, can expand per model)
MODEL_PARAMS = {
    "heart_disease_prediction": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
}