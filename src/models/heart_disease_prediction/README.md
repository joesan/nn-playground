# Heart Disease Prediction

Predict whether a patient might experience a heart problem in the next year.  
This model uses the dataset from [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

---

## Table of Contents

- [Installation](#installation)  
- [Running Inference](#running-inference)  
  - [Locally (Python)](#locally-python)  
  - [Using Docker](#using-docker)  
- [Testing the API](#testing-the-api)  
- [Expected Output](#expected-output)  
- [Development Notes](#development-notes)  

---

## Installation

Ensure you have Python 3.10+ installed.  
Create a virtual environment and install dependencies (from the project root):

```bash
python -m venvs .venv_heart_disease_prediction
source venvs/.venv_heart_disease_prediction/bin/activate
pip install -r src/models/heart_disease_prediction/requirements.txt
```

## Run Tests

To run the pytest unit tests locally, 

```bash
python -m pytest src/tests/heart_disease_prediction/test_cleanse_raw_data.py
```


## Run Inference

### Locally (Python)

From the project root:

```bash
PYTHONPATH=$(pwd)/src python -m models.heart_disease_prediction.predict_app
```

### Using Docker

First build the Docker Image 

#### Linux (VPS):

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.heart_disease_prediction \
  -t heart-disease-prediction-api:linux .
```

#### Mac:

```bash
docker build -f docker/Dockerfile.heart_disease_prediction -t heart-disease-prediction-api:mac .
```

#### Run the Docker Container

```bash
docker run -p 5000:5000 heart-disease-prediction-api:mac
```

The Flask web app will start on http://127.0.0.1:5000.

## Testing the API

Once the web app is running, test the /predict endpoint using curl:

```bash
$ curl -X POST http://127.0.0.1:5000/predict      -H "Content-Type: application/json"      -d '{
           "age": 63,
           "sex": 1,
           "cp": 3,
           "trestbps": 145,
           "chol": 233,
           "fbs": 1,
           "restecg": 0,
           "thalach": 150,
           "exang": 0,
           "oldpeak": 2.3,
           "slope": 0,
           "ca": 0,
           "thal_normal": false,
           "thal_reversible": true
         }'
```

Expected output:

```bash
{
  "prediction": 0,
  "probability": 0.18506596982479095
}
```

## Development Notes

#### Local development:

 - Use PROJECT_ROOT pointing to nn-playground (set automatically via relative paths in shared/env.py).

 - Use PYTHONPATH=$(pwd)/src so Python can locate the shared module.

#### ocker development:

 - Docker sets PROJECT_ROOT=/app via environment variable.

 - PYTHONPATH=/app ensures imports from shared work.

 - Copy the trained model into /app/models/ so the Flask app can load it.

 - This setup ensures that the code runs both locally on Mac and in Docker for Linux VPS deployments.