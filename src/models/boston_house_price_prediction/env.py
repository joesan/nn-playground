import os
from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv


# Define the filename for the resulting model
model_filename = 'boston_housing_price_predictor.pkl'
boston_dataset_url = "http://lib.stat.cmu.edu/datasets/boston"
boston_dataset = "boston_dataset.csv"


# Load environment variables from the .env file
env_file = find_dotenv(".env")
environ = load_dotenv(env_file)

# Access the environment variables
project_root = os.environ.get('PROJECT_ROOT_FOLDER')
if not project_root:
    raise ValueError("PROJECT_ROOT_FOLDER environment variable is not set.")

absolute_path = os.path.abspath(project_root)
data_dir = Path(absolute_path) / 'data' / 'raw' / 'boston_housing_price'
data_dir = data_dir.resolve()  # Resolve to get the absolute path
models_dir = Path(absolute_path) / 'models' / 'boston_housing_price'
print('***************** LOAD ENVIRONMENT ********************+')
print("Project Root DIR", project_root)
print("Project Root DIR", absolute_path)
print("Project Data DIR", data_dir)
print("Models Dump DIR", models_dir)
print('***************** LOAD ENVIRONMENT ********************+')
model_path = models_dir / model_filename