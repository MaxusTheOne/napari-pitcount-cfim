import joblib
import numpy as np
from pathlib import Path
from typing import Union

from matplotlib import pyplot as plt

from napari_pitcount_cfim.model_training.load_czi_image_and_label import czi_to_numpy
from napari_pitcount_cfim.model_training.prepare_training_data import extract_vgg_features
from training_package.prepare_training_data import extract_vgg_features as extract_vgg_features_tp

default_config = {
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "max_depth": 20,
    "n_jobs": 2,
    "random_seed": 42,
    "verbosity": 2,
    "resize_to": (256, 256),
    "channel_index": 1,
    "skip_existing": True,
    "input_dir": Path(__file__).parent / "training_data",
    "output_dir": Path(__file__).parent / "training_data" / "processed",
    "dry_run": False,

    "test_predict_path" : Path(__file__).parent / "training_data" / "TubeImage.czi",
}


class MLUser:
    """
    MLUser loads a trained model and predicts label masks from input images.
    """
    def __init__(self, ml_settings: dict = None):
        if ml_settings is None:
            ml_settings = {
                "debug": False,
                "resize_to": (256, 256),
                "model_path": None,
                "model": "modelv1",
            }
        self.settings = ml_settings

        if self.settings["model"] == "modelv1":
            self.settings["model_path"] = Path(__file__).parent / "models" / "modelv1.joblib"
            self.settings["resize_to"] = (1024, 1024)
        elif self.settings["model"] != "custom":
            raise ValueError(f"Unknown model: {self.settings['model']}. Use 'modelv1' or 'custom'.")

        model_path = self.settings["model_path"]
        if self.settings["debug"]:
            print(f"🔍 Loading model from {model_path}")

        self.clf = joblib.load(model_path)
        # number of features the model was trained on
        self.n_features = self.clf.n_features_in_
        if self.settings["debug"]:
            print(f"🔍 Loaded model from {model_path}")
            print(f"Model expects {self.n_features} features per pixel")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict label mask for a given image array.

        Parameters:
            image: np.ndarray, grayscale image array

        Returns:
            mask: np.ndarray of shape (H, W) with predicted labels
        """
        # Extract deep features
        # X_full = extract_vgg_features(image, resize_to=self.settings["resize_to"])
        X_full = extract_vgg_features_tp(image, resize_to=self.settings["resize_to"])
        # Ensure feature dimensions match model
        C = X_full.shape[2]
        if C < self.n_features:
            raise ValueError(f"Model expects {self.n_features} features but input has {C}")
        if C > self.n_features:
            X_full = X_full[..., :self.n_features]
        H, W, _ = X_full.shape
        # Flatten for prediction
        X = X_full.reshape(-1, self.n_features)
        y_pred = self.clf.predict(X)
        # Reshape to image mask
        mask = y_pred.reshape(H, W).astype(np.uint8)
        return mask
