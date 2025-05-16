import json
from os import mkdir
from pprint import pprint
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path
import numpy as np
import joblib
import random

# --- CONFIGURATION ---
CONFIG: dict[str, Any] = {
    "feature_source": "reduced",
    "feature_limit": 50,             # Use only first N VGG features (e.g. 2/128) | None = all | Affects RAM
    "max_images": 10,               # Use only first N image pairs (None = all) | Affects RAM
    "n_estimators": 50,             # Number of trees in Random Forest | Default: 50 | Affects RAM
    "max_depth": 20,                # Max depth of each tree (None = unlimited, but memory-heavy) | Default: 20 | Affects RAM
    "n_jobs": 2,                    # CPU cores to use (-1 = all, 1 = single-threaded) | Default: 2 | Affects RAM
    "verbosity": 1,                 # Verbosity level (0 = silent, 1 = some output, 2 = detailed) | Default: 0 | Affects CPU
    "random_seed": 42,
    "model_name": "rf_model",       # Model name for saving | Default: "rf_model"
}


# --- Paths ---
PROCESSED_DIR = Path(__file__).parent / "training_data" / "processed"

def get_uuids(processed_dir: Path, feature_source: str) -> list[str]:
    """
    List UUIDs based on feature files in processed_dir/feature_source.
    """
    src_dir = processed_dir / feature_source
    files = list(src_dir.glob("*.npy"))
    uuids = sorted({f.stem.split("_")[0] for f in files})
    random.seed(CONFIG["random_seed"])
    random.shuffle(uuids)
    if CONFIG["max_images"] is not None:
        uuids = uuids[: CONFIG["max_images"]]
    return uuids


def split_uuids(uuids: list[str], train_ratio: float = 2/3):
    k = int(len(uuids) * train_ratio)
    return uuids[:k], uuids[k:]


def load_dataset(
    uuids: list[str], processed_dir: Path, feature_source: str
):
    """
    Load feature and mask arrays for given UUIDs.
    """
    X_list, y_list = [], []
    feat_dir = processed_dir / feature_source
    mask_dir = processed_dir / 'masks'

    for uid in uuids:
        # Feature file
        feat_files = list(feat_dir.glob(f"{uid}_*.npy"))
        if len(feat_files) != 1:
            print(f"⚠️ Skipping {uid}: found {len(feat_files)} feature files in {feat_dir}")
            continue
        feat_path = feat_files[0]
        X = np.load(feat_path)

        # Mask file
        mask_path = mask_dir / f"{uid}_mask.npy"
        if not mask_path.exists():
            print(f"⚠️ Skipping {uid}: mask not found at {mask_path}")
            continue
        y = np.load(mask_path)

        # Feature limit
        if CONFIG["feature_limit"] is not None and X.ndim == 3:
            X = X[..., :CONFIG["feature_limit"]]

        # Flatten
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.flatten()

        if X_flat.shape[0] != y_flat.shape[0]:
            print(f"⚠️ Skipping {uid}: size mismatch {X_flat.shape[0]} vs {y_flat.shape[0]}")
            continue

        X_list.append(X_flat)
        y_list.append(y_flat)

    if not X_list:
        return np.empty((0,0)), np.empty((0,))
    return np.vstack(X_list), np.concatenate(y_list)



def train_rf_classifier(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        class_weight="balanced",
        n_jobs=CONFIG["n_jobs"],
        random_state=CONFIG["random_seed"],
        verbose=CONFIG["verbosity"]
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    if CONFIG["verbosity"] > 0:
        print(f"🔍 Evaluating model on test set with {len(X_test)} samples")
    y_pred = clf.predict(X_test)
    print("\n📊 Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, digits=4))

    if CONFIG.get("show_prediction_in_eval"):
        _plot_prediction(X_test[0], y_test[0], y_pred[0])


def train_model(config: dict = None):
    # Override module CONFIG if custom settings provided
    if config:
        CONFIG.update({
            "feature_limit": 128,
            "max_images": config.get("max_images", CONFIG.get("max_images")),
            "n_estimators": config.get("n_components", CONFIG.get("n_estimators")),
            "max_depth": config.get("max_depth", CONFIG.get("max_depth")),
            "n_jobs": config.get("n_jobs", CONFIG.get("n_jobs")),
            # 'verbosity' key maps to 'verbosity'
            "verbosity": config.get("verbosity", CONFIG.get("verbosity")),
            "random_seed": config.get("random_seed", CONFIG.get("random_seed")),
            "model_name": config.get("model_name", CONFIG.get("model_name")),
            "out_dir": config.get("output_dir", PROCESSED_DIR),
            "models_dir": config.get("models_dir", Path(__file__).parent / "models"),
            "feature_source": config.get("feature_source", "reduced"),

        })
    output_dir = CONFIG["out_dir"]
    feature_source = CONFIG["feature_source"]

    models_dir = CONFIG["models_dir"]
    model_folder = config.get("model_folder", models_dir / CONFIG["model_name"])
    model_folder.mkdir(parents=True, exist_ok=True)
    if config.get("verbosity") > 1:
        print(f"CONFIG:")
        pprint(CONFIG)
    uuids = get_uuids(output_dir, feature_source)
    train_ids, test_ids = split_uuids(uuids)

    if CONFIG["verbosity"]:
        print(f"📂 Using '{feature_source}' features from {output_dir / feature_source}")
        print(f"🎲 Training on {len(train_ids)}, testing on {len(test_ids)} samples")

    X_train, y_train = load_dataset(train_ids, output_dir, feature_source)
    X_test, y_test = load_dataset(test_ids, output_dir, feature_source)

    if CONFIG["verbosity"]:
        print(f"✅ Loaded train shape {X_train.shape}, test shape {X_test.shape}")


    clf = train_rf_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

    # Save model and metadata
    model_path = model_folder / (CONFIG["model_name"] + ".joblib")
    joblib.dump(clf, model_path)
    meta = {k: CONFIG[k] for k in ["feature_source", "feature_limit", "n_estimators", "n_components", "max_depth", "random_seed", "size_to"]}
    with open(model_folder / (CONFIG["model_name"] + "_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    if CONFIG["verbosity"]:
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Metadata saved to {model_folder}")

def _plot_prediction(image, label_mask, pred_mask):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(label_mask, cmap="gray")
    ax[1].set_title("Label Mask")
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    plt.show()

if __name__ == "__main__":
    # Limit to first 2 features for testing
    train_model()

