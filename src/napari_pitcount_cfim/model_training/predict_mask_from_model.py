import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---
DATA_PATH = Path(__file__).parent / "data"
FEATURE_PATH = DATA_PATH / "deep_features.npy"
MODEL_PATH = DATA_PATH / "rf_model.joblib"
OUTPUT_MASK_PATH = DATA_PATH / "predicted_mask.png"

def predict_mask():
    # Load deep features and model
    X_full = np.load(FEATURE_PATH)  # (H, W, C)
    H, W, C = X_full.shape
    X = X_full.reshape(-1, C)

    clf = joblib.load(MODEL_PATH)
    y_pred = clf.predict(X)

    # Reshape prediction back to image shape
    mask = y_pred.reshape(H, W).astype(np.uint8)

    # Display
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')
    plt.show()

    # Optionally save
    from imageio import imwrite
    imwrite(OUTPUT_MASK_PATH, mask * 255)  # scale to [0, 255] for PNG
    print(f"✅ Mask saved to {OUTPUT_MASK_PATH}")

if __name__ == "__main__":
    predict_mask()
