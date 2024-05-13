import os
from pathlib import Path

from dotenv import load_dotenv


# Define the terms that we want to use for search
searches = 'forest', 'bird'
# Define the filename for the resulting model
model_filename = 'is_it_a_bird_model.pkl'


# Load environment variables from the .env file
load_dotenv()


# Access the environment variables
project_root = os.environ.get('PROJECT_ROOT')
data_dir = Path(os.path.join(project_root, 'data/raw/bird_or_not'))
models_dir = Path(os.path.join(project_root, 'models/bird_or_not'))
print('***************** LOAD ENVIRONMENT ********************+')
print("Project Root DIR", project_root)
print("Project Data DIR", data_dir)
print("Models Dump DIR", models_dir)
print('***************** LOAD ENVIRONMENT ********************+')
model_path = models_dir / model_filename