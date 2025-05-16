import os
import logging
from pathlib import Path
# Configure logging format and default level (INFO). Users can adjust level via an argument or manually.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

import numpy as np
from joblib import load
from imageio import imread
import cv2
import json
from training_package.czi_transformer import czi_to_fmap
try:
    import czifile  # for reading .czi image files
except ImportError:
    raise ImportError("Please install the 'czifile' package to read .czi files")

try:
    from skimage.measure import label
except ImportError:
    raise ImportError("Please install 'scikit-image' for connected component analysis")



# For simplicity, using fixed paths (could be replaced by args as shown above):
DIR_PATH = Path(__file__).parent  # Directory of this script
Model_FOLDER_PATH = DIR_PATH / "models" / "rf_model_50_2"
MODEL_PATH = Model_FOLDER_PATH / "rf_model_50.joblib"
IMAGES_DIR = DIR_PATH / "test_data" / "Images"
LABELS_DIR = DIR_PATH / "test_data" / "Labels"
TRANSFORMER_PATH = Model_FOLDER_PATH / "transformer_50.joblib"

# Load the trained classification model (e.g., RandomForest, SVM, etc. saved via joblib)
model = load(MODEL_PATH)
logging.info(f"Loaded model from {MODEL_PATH}")

# Load the transformer
transformer = load(TRANSFORMER_PATH)

# Metrics accumulators
ious = []    # list of IoU for each image
dices = []   # list of Dice for each image
count_errors = []  # list of (pred_count - true_count) for each image



RESIZE_TO = (1024, 1024)
# Iterate over each image file in the images directory
for filename in os.listdir(IMAGES_DIR):
    if not filename.lower().endswith((".czi", ".png", ".tif", ".tiff")):
        continue  # skip non-image files
    image_path = os.path.join(IMAGES_DIR, filename)
    label_uuid = os.path.splitext(filename)[0]  # base name without extension
    # Determine the corresponding label file path (try common image extensions and .czi)
    label_path = None
    for ext in [".tif", ".tiff", ".png", ".jpg", ".czi"]:
        test_path = os.path.join(LABELS_DIR, label_uuid + ext)
        if os.path.exists(test_path):
            label_path = test_path
            break
    if label_path is None:
        logging.error(f"No label mask found for image {filename}, skipping this image")
        continue

    # Load image using czifile (for .czi) or other methods for other formats
    feat_map, image_npy = czi_to_fmap(czi_path=image_path, vgg_input_size=RESIZE_TO)

    H, W, _ = feat_map.shape
    flat = feat_map.reshape(-1, 128)
    transform_flat = transformer.transform(flat)

    # Predict
    pred_flat = model.predict(transform_flat)  # shape: (H*W,)
    pred_map = pred_flat.reshape(H, W)  # reshape to 2D prediction map

    # Upsample prediction to original size if needed (here, H,W already match original image)
    if (H, W) != (image_npy.shape[0], image_npy.shape[1]):
        # If feature map is smaller, resize the prediction mask to the input image size
        pred_mask = cv2.resize(pred_map.astype(np.uint8),
                               (image_npy.shape[1], image_npy.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    else:
        pred_mask = pred_map.astype(np.uint8)

    # Load the ground truth mask (binary image)
    if label_path.lower().endswith(".czi"):
        gt_array = czifile.imread(label_path)
        gt_array = np.squeeze(gt_array)
    else:
        gt_array = imread(label_path)
    gt_mask = (gt_array > 0).astype(np.uint8)
    if gt_mask.ndim > 2:
        gt_mask = gt_mask[..., 0]  # use the first channel if mask has multiple channels

    # Compute IoU and Dice
    intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
    iou = 1.0 if union == 0 else intersection / union
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    dice = 1.0 if (pred_sum + gt_sum) == 0 else (2.0 * intersection) / (pred_sum + gt_sum)

    # Count connected components (objects) in prediction and ground truth
    pred_labeled, pred_count = label(pred_mask, connectivity=2, return_num=True)
    gt_labeled, gt_count = label(gt_mask, connectivity=2, return_num=True)
    count_error = pred_count - gt_count

    # Log metrics for this image
    logging.debug(f"{filename}: IoU={iou:.4f}, Dice={dice:.4f}, TrueCount={gt_count}, "
                  f"PredCount={pred_count}, CountError={count_error:+d}")

    # Accumulate metrics
    ious.append(iou);
    dices.append(dice);
    count_errors.append(count_error)

if len(ious) == 0:
    logging.error("No images were processed – please check the input paths.")
else:
    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    mae_count = float(np.mean(np.abs(count_errors)))   # Mean Absolute Error of counts
    bias_count = float(np.mean(count_errors))          # Average (predicted - true) count

    # Log summary metrics
    logging.info(f"Mean IoU (Jaccard Index): {mean_iou:.4f}")
    logging.info(f"Mean Dice Coefficient: {mean_dice:.4f}")
    logging.info(f"Mean Absolute Count Error (MAE): {mae_count:.2f}")
    logging.info(f"Count Bias (Pred - True): {bias_count:.2f}")

    # Save evaluation statistics to the model folder
    stats = {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mae_count": mae_count,
        "bias_count": bias_count
    }
    stats_path = MODEL_PATH.parent / "evaluation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logging.info(f"Saved evaluation stats to {stats_path}")


