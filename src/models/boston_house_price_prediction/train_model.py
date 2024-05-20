from pathlib import Path

import numpy as np
import pandas as pd
import zipfile
from pandas import DataFrame
from io import StringIO
from colorist import red, Color, BrightColor

from src.models.boston_house_price_prediction import env
import os
import requests
from urllib.request import urlretrieve


def load_data(data_dir=env.data_dir):
    # Check if the dataset file exists in the data directory
    dataset_file = Path(data_dir / env.boston_dataset).resolve()
    print(f"Dataset file: {Color.GREEN}{dataset_file}{Color.OFF}")

    # We get the column names after inspecting the boston housing dataset visually
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                    'MEDV']

    if os.path.exists(dataset_file):
        # If the dataset file exists, load it directly
        print("Dataset file exists.")
        raw_df = pd.read_csv(dataset_file, header=None, delimiter=r"\s+", names=column_names, na_values='?')
    else:
        print("Downloading Dataset......")
        urlretrieve(env.boston_dataset_url, Path(data_dir) / 'boston_housing.csv')
        raw_df = pd.read_csv(dataset_file, header=None, delimiter=r"\s+", names=column_names)
    return raw_df

