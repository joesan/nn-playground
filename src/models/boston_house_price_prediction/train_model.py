from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from io import StringIO

from src.models.boston_house_price_prediction import env
import os
import requests


# https://lib.stat.cmu.edu/datasets/boston
def load_data(data_dir):
    print(data_dir)
    # Check if the dataset file exists in the data directory
    dataset_file = Path(os.path.join(data_dir, env.boston_dataset))
    print(dataset_file)
    if os.path.exists(dataset_file):
        # If the dataset file exists, load it directly
        raw_df = pd.read_csv(dataset_file, sep="\s+", skiprows=22, header=None)
    else:
        # If the dataset file doesn't exist, fetch it from the URL
        response = requests.get(env.boston_dataset_url)
        if response.status_code == 200:
            # Parse the CSV data from the response content
            csv_data = response.text
            raw_df = pd.read_csv(StringIO(csv_data), sep="\s+", skiprows=22, header=None)
            # Save the dataset to the data directory for future use
            raw_df.to_csv(dataset_file, index=False)
        else:
            print("Failed to fetch data from URL:", env.boston_dataset_url)
            return None
    return raw_df

