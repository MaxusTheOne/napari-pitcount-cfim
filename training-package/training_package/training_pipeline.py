"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from pathlib import Path
from training_package.prepare_training_data import process_all
from training_package.train_model_from_dataset import train_model

deep_training_config = {
    "feature_limit": 128,
    "max_images": None,
    "n_estimators": 300,
    "max_depth": None,
    "n_jobs": -1,

    "resize_to": None,

    "skip_existing": False,

    }
light_training_config = {
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "max_depth": 20,
    "n_jobs": -1,
    "random_seed": 42,
    "resize_to": (1024, 1024),
}

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

def prepare_data_and_train(config: dict):
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
            verbosity: int
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

    if config["verbosity"] > 0:
        print("✅ Training pipeline completed successfully.")

    if config.get("test_predict_path") is not None:
        from training_package.predict_mask_from_model import predict_mask
        predict_mask(czi_path=config["test_predict_path"])
        if config["verbosity"] > 0:
            print("✅ Prediction completed successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a RandomForest model for pit counting.")
    parser.add_argument("--config_index", type=int, default=1, help="0 = deep, 1 = light")
    args = parser.parse_args()

    config_pick = [deep_training_config, light_training_config][args.config_index]
    config = {**default_config, **config_pick}

    print(f"Training with config: {config}")
    prepare_data_and_train(config)


