import os
from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv


class HeartDiseasePredictionConfig:
    def __init__(self, env_file_path=".env", model_filename='heart_disease_prediction.pkl'):
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
        self.data_dir = Path(self.absolute_path) / 'data' / 'raw' / 'heart_disease_dataset'
        self.data_dir = self.data_dir.resolve()  # Resolve to get the absolute path
        self.models_dir = Path(self.absolute_path) / 'models' / 'heart_disease_prediction'
        self.model_path = self.models_dir / self.model_filename

        self._print_environment_details()

    def _print_environment_details(self):
        print('***************** LOAD ENVIRONMENT ********************')
        print("Project Root DIR:", self.absolute_path)
        print("Project Data DIR:", self.data_dir)
        print("Models Dump DIR:", self.models_dir)
        print('***************** LOAD ENVIRONMENT ********************')


# Example usage:
#config = HeartDiseasePredictionConfig()
#config.load_environment()
