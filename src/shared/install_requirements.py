import subprocess
from src.shared.env import *


def install_requirements_txt(model_name=None):
    """
    Installs Python dependencies for the project, with optional model-specific requirements.

    Args:
        model_name (str, optional): Name of the model folder inside `src/models`.
            If provided, the function will first try to install dependencies from:
                `src/models/<model_name>/requirements.txt`.
            If that file does not exist, it falls back to the global `requirements.txt`.

    Behavior:
        1. If `model_name` is given and a corresponding requirements file exists, install it.
        2. Otherwise, install the global project-level requirements.txt.
    """
    # Determine the requirements file path
    if model_name:
        model_req_path = os.path.join(PROJECT_ROOT, "src", "models", model_name, "requirements.txt")
        if os.path.exists(model_req_path):
            print(f"Installing dependencies from {model_req_path}")
            subprocess.check_call(['pip', 'install', '-r', model_req_path])
            return
        else:
            print(f"No model-specific requirements found for '{model_name}', falling back to global requirements.")

    # Fallback to global requirements
    global_req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.exists(global_req_path):
        print(f"Installing dependencies from global requirements {global_req_path}")
        subprocess.check_call(['pip', 'install', '-r', global_req_path])
    else:
        raise FileNotFoundError("No requirements.txt found globally or for the specified model.")