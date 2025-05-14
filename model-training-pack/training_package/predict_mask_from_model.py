import json

import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from training_package.load_czi_image_and_label import czi_to_numpy
from training_package.extract_deep_50 import extract_deep_features_50
import argparse

FEATURE_LIMIT = 2

def predict_mask(image, clf, output_path, visualize=True, resize_to=(1024, 1024)):
    # Extract deep features
    X_full = extract_deep_features_50(image, resize_to=resize_to)

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
    values, counts = np.unique(mask, return_counts=True)
    print(dict(zip(values, counts)))
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
        overlay = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(image, cmap='gray')
        ax2.imshow(overlay, cmap='Reds_r', alpha=1)
        ax2.set_title("Image with Predicted Mask Overlay")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    save_mask(mask, output_path)


def save_mask(mask, output_path):
    # Optionally save
    from imageio import imwrite
    imwrite(output_path, mask * 255)
    print(f"✅ Mask saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict pit masks from a trained RF model.")
    parser.add_argument(
        "--czi", "-i",
        type=Path,
        required=True,
        help="Path to the .czi image you want to run. \n If a directory is given, it will process all .czi files in it."
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=Path.cwd() / "models",
        help="Path to the dir of the trained .joblib model."
    )
    parser.add_argument(
        "--model-name", "-mn",
        type=str,
        default="rf_model",
        help="Name of the trained model file (without extension)."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.cwd() / "predicted_mask.png",
        help="Where to save the output mask PNG."
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization."
    )
    args = parser.parse_args()

    model_dir = args.model
    model_name = args.model_name
    model_file = model_dir / (model_name + ".joblib")
    print(f"Loading meta from {model_dir / (model_name + 'metadata.json')}")
    with open(model_dir / (model_name + "metadata.json")) as f:
        meta = json.load(f)
        print(f"Meta: {meta}")

    clf = joblib.load(model_file)

    if args.czi.is_dir():
        czi_paths = sorted(args.czi.glob("*.czi"))
    else:
        czi_paths = [args.czi]

        # Process each file
    for czi_path in czi_paths:
        print(f"Processing {czi_path}")
        image = czi_to_numpy(czi_path)
        # Determine output path
        if args.czi.is_dir():
            out_dir = args.output
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (czi_path.stem + "_mask.png")
        else:
            output_path = args.output

        predict_mask(
            image,
            clf,
            output_path,
            visualize=not args.no_viz,
            resize_to=tuple(meta["resize_to"])
        )

    return

if __name__ == "__main__":
    main()
