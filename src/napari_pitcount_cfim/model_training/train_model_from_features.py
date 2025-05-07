import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

# --- Config ---
DATA_PATH = Path(__file__).parent / "data"
FEATURE_PATH = DATA_PATH / "deep_features.npy"
LABEL_PATH = DATA_PATH / "label.npy"
MODEL_PATH = DATA_PATH / "rf_model.joblib"
SAMPLE_RATIO = 3  # Neg:Pos class ratio

def train_model():
    # --- Load ---
    X_full = np.load(FEATURE_PATH)         # Shape: (H, W, C)
    y_full = np.load(LABEL_PATH)          # Shape: (H, W)

    H, W, C = X_full.shape
    X = X_full.reshape(-1, C)
    y = y_full.reshape(-1)

    # --- Balance classes ---
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.random.choice(np.where(y == 0)[0], size=len(pos_indices) * SAMPLE_RATIO, replace=False)
    sample_indices = np.concatenate([pos_indices, neg_indices])

    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=42
    )

    # --- Train Random Forest ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"🎯 Precision: {precision:.4f}")
    print(classification_report(y_test, y_pred))

    # --- Save model ---
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()