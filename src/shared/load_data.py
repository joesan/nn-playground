import os
import requests
import pandas as pd
from src.shared import env


def download_csv(csv_name: str) -> pd.DataFrame:
    """
    Loads a CSV into a DataFrame. Downloads if missing locally.

    Args:
        csv_name (str): Key for CSV in env.CSV_URLS dict.

    Returns:
        pd.DataFrame
    """
    # Get URL and local path from env
    url = env.CSV_RAW_DATASET_URLS[csv_name]
    local_path = env.HEART_DISEASE_CSV if csv_name == "heart_disease" else os.path.join(env.DATA_RAW_DIR, f"{csv_name}.csv")

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download if not exists
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading {csv_name} CSV from {url} to {local_path}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Download complete.")
    else:
        print(f"[INFO] {csv_name} CSV already exists at {local_path}, skipping download.")

    # Load DataFrame
    df = pd.read_csv(local_path)
    print(f"[INFO] CSV loaded with shape {df.shape}.")
    return df