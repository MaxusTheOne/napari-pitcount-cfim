import tensorflow as tf
import keras
from pathlib import Path

model_file = Path(__file__).resolve().parent / "spec_out" / "modelid=ac152c5c-b616-431c-ae2a-ea0ec307ce76"

keras_model = keras.layers.TFSMLayer(model_file, call_endpoint='serving_default')

# Create an inference-only layer using the legacy model folder.
tfs_model_layer = keras.layers.TFSMLayer(model_file, call_endpoint='serving_default')
print("Inference layer created successfully!")