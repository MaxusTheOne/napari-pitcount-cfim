import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from training_package.load_czi_image_and_label import czi_to_numpy
from training_package.extract_deep_features import extract_vgg_features

# --- Config ---
DATA_PATH = Path(__file__).parent / "training_data"
CZI_PATH = DATA_PATH / "TubeImage.czi"  # update or pass as argument
MODEL_PATH = DATA_PATH / "processed" / "rf_model.joblib"
OUTPUT_MASK_PATH = DATA_PATH / "predicted_mask.png"
FEATURE_LIMIT = 2

def predict_mask(czi_path=CZI_PATH, model_path=MODEL_PATH, output_path=OUTPUT_MASK_PATH, visualize=True):
    # Load image and extract deep features
    image = czi_to_numpy(czi_path)
    X_full = extract_vgg_features(image)

    # Load model
    clf = joblib.load(model_path)
    # Ensure feature dimension matches model expectations
    n_features = clf.n_features_in_
    if X_full.shape[2] < n_features:
        raise ValueError(f"Model expects {n_features} features but input has {X_full.shape[2]}")
    if X_full.shape[2] > n_features:
        # truncate excess features
        X_full = X_full[..., :n_features]
    H, W, C = X_full.shape
    X = X_full.reshape(-1, C)

    # Predict
    y_pred = clf.predict(X)

    # Reshape prediction back to image shape
    mask = y_pred.reshape(H, W).astype(np.uint8)

    # Display
    if visualize:
        # Display original image and overlay of predicted mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1, ax2 = axes
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        # Overlay mask on image
        ax2.imshow(image, cmap='gray')
        overlay = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(overlay, cmap='jet', alpha=0.5)
        ax2.set_title("Image with Predicted Mask Overlay")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    # Optionally save
    from imageio import imwrite
    imwrite(output_path, mask * 255)  # scale to [0, 255] for PNG
    print(f"✅ Mask saved to {output_path}")

if __name__ == "__main__":
    predict_mask()

