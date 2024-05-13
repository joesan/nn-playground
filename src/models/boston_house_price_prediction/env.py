import os
from pathlib import Path

from dotenv import load_dotenv


# Define the filename for the resulting model
model_filename = 'boston_housing_price_predictor.pkl'


# Load environment variables from the .env file
load_dotenv()


# Access the environment variables
project_root = os.environ.get('PROJECT_ROOT')
data_dir = Path(os.path.join(project_root, 'data/raw/boston_housing_price'))
models_dir = Path(os.path.join(project_root, 'models/boston_housing_price'))
print('***************** LOAD ENVIRONMENT ********************+')
print("Project Root DIR", project_root)
print("Project Data DIR", data_dir)
print("Models Dump DIR", models_dir)
print('***************** LOAD ENVIRONMENT ********************+')
model_path = models_dir / model_filename