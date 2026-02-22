from tensorflow import keras
from tensorflow.keras import layers
from src.shared import env


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


def train_model(model, X_train, y_train, X_test, y_test, model_name: str):
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
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=params.get("epochs", 50),
        batch_size=params.get("batch_size", 16),
        verbose=1
    )

    # --------------------------
    # 3. Save the model globally
    # --------------------------
    os.makedirs(env.GLOBAL_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(env.GLOBAL_MODELS_DIR, f"{model_name}.keras")
    model.save(model_path)

    print(f"[INFO] Model saved to {model_path}")

    return model, history
