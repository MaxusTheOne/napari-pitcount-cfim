import xml.etree.ElementTree as ET
from pathlib import Path
import tensorflow as tf
from czmodel.tensorflow import LegacyConverter
from czmodel.core.legacy_model_metadata import ModelMetadata as LegacyModelMetadata
import keras

original_from_xml = LegacyModelMetadata.from_xml
# Patch XML metadata if needed
def patched_from_xml(xml_file: str) -> LegacyModelMetadata:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if root.find('BorderSize') is None:
        default_elem = ET.Element('BorderSize')
        default_elem.text = '90'
        root.append(default_elem)
        tree.write(xml_file)
    return original_from_xml(xml_file)

LegacyModelMetadata.from_xml = patched_from_xml


def load_converted_model(czseg_file_path: Path, spec_out_path: Path, input_shape: tuple,
                         call_endpoint: str = 'serving_default') -> tf.keras.Model:
    # Convert legacy model and get legacy weights path.
    converter = LegacyConverter()
    model_metadata, legacy_weights_file = converter.unpack_model(
        model_file=str(czseg_file_path),
        target_dir=spec_out_path
    )
    print("Metadata extracted:", model_metadata)
    print(f"Debug | Legacy weights path: {legacy_weights_file}")

    # Check if the legacy file is a proper SavedModel directory.
    saved_model_pb = Path(legacy_weights_file) / "saved_model.pb"
    if not saved_model_pb.exists():
        raise OSError(f"Legacy weights file is not a SavedModel: {legacy_weights_file}")

    # Create an inference model using TFSMLayer with legacy weights.
    tfs_layer = keras.layers.TFSMLayer(str(legacy_weights_file), call_endpoint=call_endpoint)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = tfs_layer(inputs)
    temp_model = tf.keras.Model(inputs, outputs)

    # Save weights to a supported file (e.g. .weights.h5).
    supported_weights_path = spec_out_path / 'converted.weights.h5'
    temp_model.save_weights(str(supported_weights_path))
    print(f"Supported weights saved to: {supported_weights_path}")

    # Rebuild the inference model and load the supported weights.
    inputs = tf.keras.Input(shape=input_shape)
    outputs = tfs_layer(inputs)
    new_model = tf.keras.Model(inputs, outputs)
    new_model.load_weights(str(supported_weights_path))
    print("Model loaded successfully with supported weights!")
    return new_model

# Example usage:
if __name__ == "__main__":
    czseg_file = Path(r'C:\napari-pitcount-cfim\src\resources\models\Frederikke.pit.fluor.czseg')
    spec_out = Path(__file__).resolve().parent / 'spec_out'
    model = load_converted_model(czseg_file, spec_out, input_shape=(64,))