import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import random

# --- CONFIGURATION ---
CONFIG = {
    "feature_limit": 2,             # Use only first N VGG features (e.g. 2/128)
    "max_images": 10,               # Use only first N image pairs (None = all)
    "n_estimators": 50,             # Number of trees in Random Forest (smaller = less RAM)
    "max_depth": 20,                # Max depth of each tree (None = unlimited, but memory-heavy)
    "n_jobs": 2,                    # CPU cores to use (-1 = all, 1 = single-threaded)
    "random_seed": 42
}


# --- Paths ---
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "rf_model.joblib"


def get_uuid_split(train_ratio=2 / 3):
    uuids = sorted([p.name for p in PROCESSED_DIR.iterdir() if (p / "features.npy").exists()])
    random.seed(CONFIG["random_seed"])
    random.shuffle(uuids)
    if CONFIG["max_images"] is not None:
        uuids = uuids[:CONFIG["max_images"]]
    split_idx = int(len(uuids) * train_ratio)
    return uuids[:split_idx], uuids[split_idx:]


def load_dataset(uuid_list):
    X_list, y_list = [], []
    for uid in uuid_list:
        pair_path = PROCESSED_DIR / uid
        X = np.load(pair_path / "features.npy")
        y = np.load(pair_path / "label.npy")

        if CONFIG["feature_limit"]:
            X = X[..., :CONFIG["feature_limit"]]

        X_list.append(X.reshape(-1, X.shape[-1]))  # (H*W, F)
        y_list.append(y.flatten())  # (H*W,)

    return np.concatenate(X_list), np.concatenate(y_list)


def train_rf_classifier(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        class_weight="balanced",
        n_jobs=CONFIG["n_jobs"],
        random_state=CONFIG["random_seed"]
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\n📊 Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, digits=4))


def main(feature_limit=None):
    train_uuids, test_uuids = get_uuid_split()
    print(f"Training on {len(train_uuids)} images, testing on {len(test_uuids)}")

    X_train, y_train = load_dataset(train_uuids, feature_limit)
    X_test, y_test = load_dataset(test_uuids, feature_limit)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    clf = train_rf_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    # Limit to first 2 features for testing
    main(feature_limit=None)
