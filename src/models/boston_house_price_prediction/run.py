import train_model
import predict_model
import subprocess
import os

from src.models.is_it_a_bird import env


def install_requirements():
    # Run pip install command to install dependencies from _requirements.txt
    subprocess.check_call(['pip', 'install', '-r', os.path.join(env.project_root, '_requirements.txt')])

def run():
    # Install requirements first
    #install_requirements()
    # Run the train_model script
    train_model.download_train_dump_model()

    # Start the Flask app
    predict_model.app.run(debug=True)

if __name__ == "__main__":
    run()
