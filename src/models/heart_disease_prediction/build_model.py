from tensorflow import keras
from src.shared import env
import os
import tensorflow as tf

prefix = "    "

def build_model(input_dim: int, model_name: str) -> keras.Model:
    """
    Build a Keras model using the functional API and parameters from env.py.

    Args:
        input_dim (int): Number of input features
        model_name (str): Key inside env.MODEL_PARAMS

    Returns:
        Compiled Keras model
    """
    if model_name not in env.MODEL_PARAMS:
        raise ValueError(f"Model '{model_name}' not found in env.MODEL_PARAMS")

    params = env.MODEL_PARAMS[model_name]

    # --------------------------
    # 1. Define input layer
    # --------------------------
    inputs = keras.Input(shape=(input_dim,), name="Input")

    # --------------------------
    # 2. Hidden layers
    # --------------------------
    x = inputs
    for i, units in enumerate(params["hidden_layers"]):
        x = keras.layers.Dense(
            units,
            activation=params["activation"],
            name=f"Hidden_{i+1}"
        )(x)

    # --------------------------
    # 3. Output layer
    # --------------------------
    outputs = keras.layers.Dense(
        1,
        activation=params["output_activation"],
        name="Output"
    )(x)

    # --------------------------
    # 4. Build model
    # --------------------------
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    # --------------------------
    # 5. Compile
    # --------------------------
    model.compile(
        optimizer=params["optimizer"],
        loss=params["loss"],
        metrics=params["metrics"]
    )

    print(f"{prefix} {model.summary()}")
    return model


def train_model(model, X_train, y_train, model_name: str):
    """
    Train a pre-built Keras model using parameters from env.py and save it globally.

    Args:
        model (keras.Model): Pre-built Keras model
        X_train, y_train, X_test, y_test: Training and test data
        model_name (str): Key inside env.MODEL_PARAMS (for hyperparameters like epochs/batch_size)

    Returns:
        model: Trained Keras model
        history: Training history object
    """

    # --------------------------
    # 1. Load training parameters
    # --------------------------
    if model_name not in env.MODEL_PARAMS:
        raise ValueError(f"Model '{model_name}' not found in env.MODEL_PARAMS")

    params = env.MODEL_PARAMS[model_name]

    # --------------------------
    # 2. Train the model
    # --------------------------
    history = model.fit(
        X_train,  # Features for training
        y_train,  # Target labels
        epochs = params.get("epochs", 50),  # Number of passes over the full dataset
        batch_size = params.get("batch_size", 16),  # Number of samples per gradient update
        verbose = 1,  # Print progress per epoch
        validation_split = params.get("validation_split", 0.2)  # Automatically hold out 20% of train_X/train_y for validation
    )

    # --------------------------
    # 3. Save the model globally
    # --------------------------
    print(tf.config.list_physical_devices())
    os.makedirs(env.GLOBAL_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(env.GLOBAL_MODELS_DIR, f"{model_name}.keras")
    model.save(model_path)

    print(f"[INFO] Model saved to {model_path}")

    return model, history
