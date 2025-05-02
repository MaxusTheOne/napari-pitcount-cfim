import tensorflow as tf
import numpy as np
from pathlib import Path

def load_legacy_model(model_file: Path, input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Custom loader for legacy .model files.

    Args:
        model_file (Path): Path to the legacy .model file.
        input_shape (tuple): Input shape for the model.
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Reconstructed model with loaded weights.
    """
    # Step 1: Read the .model file (assuming it's a binary file with weights).
    if not model_file.exists():
        raise FileNotFoundError(f"Legacy model file not found: {model_file}")

    print(f"Loading legacy model from: {model_file}")
    with open(model_file, "rb") as f:
        # Example: Deserialize weights (adjust based on actual file format).
        weights_data = np.load(f, allow_pickle=True).item()

    # Step 2: Rebuild the model architecture.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Step 3: Load weights into the model.
    for layer, weights in zip(model.layers, weights_data.get("weights", [])):
        layer.set_weights(weights)

    print("Legacy model loaded successfully!")
    return model

# Example usage:
if __name__ == "__main__":
    legacy_model_path = Path(r"C:\napari-pitcount-cfim\src\resources\models\flour_model\ac152c5c-b616-431c-ae2a-ea0ec307ce76.model")
    model = load_legacy_model(legacy_model_path, input_shape=(64,), num_classes=2)
    print(f"Model summary:\n{model.summary()}")