import os
from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv


# Define the filename for the resulting model
#model_filename = 'boston_housing_price_predictor.pkl'
#boston_dataset_url = "http://lib.stat.cmu.edu/datasets/boston"
#boston_dataset_url = 'https://raw.githubusercontent.com/noahgift/boston_housing_pickle/master/housing.csv'
#boston_dataset = 'boston_housing.csv'


# Load environment variables from the .env file
#env_file = find_dotenv(".env")
#environ = load_dotenv(env_file)

# Access the environment variables
#project_root = os.environ.get('PROJECT_ROOT_FOLDER')
if not project_root:
    raise ValueError("PROJECT_ROOT_FOLDER environment variable is not set.")

absolute_path = os.path.abspath(project_root)
data_dir = Path(absolute_path) / 'data' / 'raw' / 'boston_housing_price'
data_dir = data_dir.resolve()  # Resolve to get the absolute path
models_dir = Path(absolute_path) / 'models' / 'boston_housing_price'
print('***************** LOAD ENVIRONMENT ********************+')
print("Project Root DIR", absolute_path)
print("Project Data DIR", data_dir)
print("Models Dump DIR", models_dir)
print('***************** LOAD ENVIRONMENT ********************+')
model_path = models_dir / model_filename


class BostonHousingConfig:
    def __init__(self, env_file_path=".env", model_filename='boston_housing_price_predictor.pkl'):
        self.env_file_path = env_file_path
        self.model_filename = model_filename
        self.project_root = None
        self.absolute_path = None
        self.data_dir = None
        self.models_dir = None
        self.model_path = None

    def load_environment(self):
        # Load environment variables from the .env file
        env_file = find_dotenv(self.env_file_path)
        if not load_dotenv(env_file):
            raise ValueError(f"Failed to load environment file: {self.env_file_path}")

        # Access the environment variables
        self.project_root = os.environ.get('PROJECT_ROOT_FOLDER')
        if not self.project_root:
            raise ValueError("PROJECT_ROOT_FOLDER environment variable is not set.")

        # Set up the paths
        self.absolute_path = os.path.abspath(self.project_root)
        self.data_dir = Path(self.absolute_path) / 'data' / 'raw' / 'boston_housing_price'
        self.data_dir = self.data_dir.resolve()  # Resolve to get the absolute path
        self.models_dir = Path(self.absolute_path) / 'models' / 'boston_housing_price'
        self.model_path = self.models_dir / self.model_filename

        self._print_environment_details()

    def _print_environment_details(self):
        print('***************** LOAD ENVIRONMENT ********************')
        print("Project Root DIR:", self.absolute_path)
        print("Project Data DIR:", self.data_dir)
        print("Models Dump DIR:", self.models_dir)
        print('***************** LOAD ENVIRONMENT ********************')


# Example usage:
config = BostonHousingConfig()
config.load_environment()
