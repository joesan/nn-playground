from flask import Flask, request, jsonify
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import joblib

app = Flask(__name__)

# Load the model
learn = None


def load_model():
    global learn
    learn = load_learner('is_it_a_bird_model.pkl')


# API endpoint for prediction
# Usage: http://localhost:5000/predict
# Example: curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load image from request
        image_file = request.files['image']
        image = PILImage.create(image_file)

        # Make prediction
        is_bird, _, probs = learn.predict(image)

        # Prepare response
        response = {
            'is_bird': bool(is_bird),
            'probability': float(probs[0])
        }

        return jsonify(response)


if __name__ == '__main__':
    # Load the model when the script is executed
    load_model()
    app.run(debug=True)

