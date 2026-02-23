## heart-disease-prediction

Predict if a patient might experience a heart problem in the next year. This uses the 
dataset from [here](https://archive.ics.uci.edu/dataset/45/heart+disease).

### Run Inference

On the project root, run the command below:

python -m src.models.heart_disease_prediction.predict_app

If using Docker, use the following command to build the Docker image locally:

For builds on Linux:

$ docker buildx build  --platform linux/amd64  -f docker/Dockerfile.heart_disease_prediction  -t heart-disease-prediction-api:linux  .

For build on Mac:

docker build -f docker/Dockerfile.heart_disease_prediction -t heart-disease-prediction-api:mac .

To run the image:

docker run -p 5000:5000 heart-disease-prediction-api:mac