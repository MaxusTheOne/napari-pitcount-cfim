"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from pathlib import Path
from training_package.prepare_training_data import process_all
from training_package.train_model_from_dataset import train_model
import argparse
deep_training_config = {
    "feature_limit": 50,
    "max_images": None,
    "n_estimators": 300,
    "max_depth": None,
    "n_jobs": -1,

    "resize_to": None,

    "skip_existing": False,

    }
light_training_config = {
    "feature_limit": 4,
    "max_images": 14,
    "n_estimators": 10,
    "max_depth": 20,
    "n_jobs": -1,
    "random_seed": 42,
    "resize_to": None,
    "skip_existing": False,
}

default_config = {
    "model_name": "rf_model.joblib",
    
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "max_depth": 20,
    "n_jobs": 2,
    "random_seed": 42,
    "verbosity": 2,
    "resize_to": (256, 256),
    "channel_index": 0,
    "skip_existing": True,

    "dry_run": False,


    # "test_predict_path" : Path(__file__).parent / "training_data" / "TubeImage.czi",
    "test_predict_path" : None,
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
    config_changes = process_all(config)
    print("🔧 Config changes:", {k: v for k, v in config_changes.items() if config.get(k) != v})
    config.update(config_changes)
    train_model(config)

    if config["verbosity"] > 0:
        print("✅ Training pipeline completed successfully.")


def main():

    parser = argparse.ArgumentParser(description="Train a RandomForest model for pit counting.")
    parser.add_argument("--config-index", type=int, default=1, help="Choose base config: 0 = deep, 1 = light")
    parser.add_argument("--config-options", type=str, help="JSON-like string for additional config options, e.g., '{\"verbose\":1, \"max_images\":null}'")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Path to input directory containing images and labels.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path.cwd() / "output", help="Path to output directory for trained model.")
    parser.add_argument("--model-dir", "-m", type=Path, default=Path.cwd() / "models", help="Path to output directory for trained model.")

    args = parser.parse_args()

    config_pick = [deep_training_config, light_training_config][args.config_index]
    config = {**default_config, **config_pick}

    config["input_dir"] = args.input_dir
    config["model_dir"] = args.model_dir
    config["output_dir"] = args.output_dir


    if args.config_options:
        import json
        config_options = json.loads(args.config_options)
        config.update(config_options)

    print(f"Training with config: {config}")
    prepare_data_and_train(config)


