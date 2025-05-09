"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from pathlib import Path
from napari_pitcount_cfim.model_training.prepare_training_data import process_all
from napari_pitcount_cfim.model_training.train_model_from_dataset import train_model

deep_training_config = {
    "feature_limit": 128,
    "max_images": None,
    "n_estimators": 300,
    "max_depth": None,
    "n_jobs": -1,
    "random_seed": 42,
    "verbose": 1,
    "resize_to": None,
    "channel_index": 1,
    "skip_existing": True,
    "output_dir": Path(__file__).parent / "training_data" / "processed",
    "dry_run": False
    }
light_training_config = {
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "max_depth": None,
    "n_jobs": -1,
    "random_seed": 42,
    "verbose": 1,
    "resize_to": (256, 256),
    "channel_index": 1,
    "skip_existing": True,
    "output_dir": Path(__file__).parent / "training_data" / "processed",
    "dry_run": False
}

default_config = {
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "max_depth": None,
    "n_jobs": -1,
    "random_seed": 42,
    "verbose": 1,
    "resize_to": (256, 256),
    "channel_index": 1,
    "skip_existing": True,
    "output_dir": Path(__file__).parent / "training_data" / "processed",
    "dry_run": False,

    "test_predict_path" : Path(__file__).parent / "training_data" / "TubeImage.czi",
}

def prepare_and_train_from_dataset(config: dict):
    """
    Run full training pipeline:
      - processes raw images
      - trains RF model with given config

    Args:
        config (dict): keys:
            feature_limit: int or None
            max_images: int or None
            n_estimators: int
            max_depth: int or None
            n_jobs: int
            verbose: int
            random_seed: int

            resize_to: tuple of int or None
            channel_index: int
            skip_existing: bool
            output_dir: Path
            dry_run: bool
    """
    # Prepare data and train
    process_all(config)
    train_model(config)

    if config["verbose"] > 0:
        print("✅ Training pipeline completed successfully.")

    if config["test_predict_path"] is not None:
        from napari_pitcount_cfim.model_training.predict_mask_from_model import predict_mask
        predict_mask(czi_path=config["test_predict_path"])
        if config["verbose"] > 0:
            print("✅ Prediction completed successfully.")

if __name__ == "__main__":
    # Example usage

    prepare_and_train_from_dataset(light_training_config)

