#!/usr/bin/env python3
"""
prepare_training_data.py

Extract deep features (128-dim) from grayscale microscopy images using VGG19 up to conv2_2,
fit a Gaussian random projection on the pooled features, and save both raw and reduced features
for downstream classifier training. Also converts binary ground-truth masks to .npy.

This module exposes a `prepare_training_data` function that accepts a single `training_settings` dict
with all configuration parameters (and reasonable defaults), suitable for invocation by a pipeline.
"""
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from joblib import dump
from sklearn.random_projection import GaussianRandomProjection
import czifile

# --- VGG19 Feature Extractor Setup ---
VGG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
vgg_conv2_2 = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
vgg_conv2_2.eval()


def load_gray(path: Path) -> np.ndarray:
    """
    Load an image (CZI or standard formats) as a 2D uint8 grayscale array.
    """
    arr = czifile.imread(str(path)) if path.suffix.lower() == '.czi' else cv2.imread(
        str(path), cv2.IMREAD_UNCHANGED
    )
    arr = np.squeeze(arr)
    # Convert multi-channel to gray
    if arr.ndim == 3:
        channels = arr.shape[-1]
        if channels in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            if rgb.max() > 0:
                rgb = rgb / rgb.max() * 255
            arr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[..., 0]
    # Normalize to [0,255]
    arr = arr.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """
    Load a binary mask (CZI or standard formats) and convert to 2D uint8 binary array (0/1).
    """
    arr = czifile.imread(str(path)) if path.suffix.lower() == '.czi' else cv2.imread(
        str(path), cv2.IMREAD_UNCHANGED
    )
    arr = np.squeeze(arr)
    # If multi-channel, take first channel
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Binarize mask
    mask = (arr > 0).astype(np.uint8)
    return mask


def extract_raw_features(
    gray_img: np.ndarray,
    vgg_input_size: tuple
) -> np.ndarray:
    """
    Extract raw VGG19 conv2_2 features for a grayscale image.

    gray_img: 2D uint8 array
    vgg_input_size: (height, width) for VGG input
    returns: H×W×128 float32 feature map
    """
    H0, W0 = gray_img.shape
    # Resize for VGG input
    resized = cv2.resize(gray_img, dsize=vgg_input_size[::-1], interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    x = VGG_TRANSFORM(rgb).unsqueeze(0)
    with torch.no_grad():
        feats = vgg_conv2_2(x)
        feats = F.interpolate(feats, size=vgg_input_size, mode='bilinear', align_corners=False)
    fmap = feats.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Upsample to original resolution if needed
    if vgg_input_size != (H0, W0):
        fmap = cv2.resize(fmap, (W0, H0), interpolation=cv2.INTER_LINEAR)
        fmap = fmap.reshape(H0, W0, -1)
    return fmap.astype(np.float32)


def prepare_training_data(training_settings: dict):
    """
    Prepare training data by extracting deep features, reducing dimensions, and saving masks.

    Required keys in training_settings dict (with defaults):
      - images_dir: Path to UUID-folders with images (must provide)
      - labels_dir: Path to UUID-folders with binary masks (must provide)
      - processed_dir: Path under which 'raw/', 'reduced/', and 'masks/' will be created (must provide)
      - models_dir: Path where transformer will be saved (must provide)
      - n_components: int, default 50
      - random_state: int, default 42
      - vgg_input_size: tuple (H, W), default (256, 256)
    Other args:
      - verbosity: int, default 1


    This function:
      1. Extracts and saves raw 128-dim feature maps per image.
      2. Converts and saves binary masks to .npy.
      3. Fits a GaussianRandomProjection on the pooled features.
      4. Saves the fitted transformer.
      5. Applies transformer to each raw map and saves the reduced features.
    """
    images_dir = Path(training_settings['input_dir']) / 'Images'
    labels_dir = Path(training_settings['input_dir']) / 'Labels'
    output_dir = Path(training_settings['output_dir'])
    models_dir = Path(training_settings['models_dir']) / training_settings['model_name']
    n_components = training_settings.get('n_components', 50)
    random_state = training_settings.get('random_state', 42)
    vgg_input_size = tuple(training_settings.get('resize_to', (256, 256)))

    max_images = training_settings.get('max_images', None)
    verbosity = training_settings.get('verbosity', 1)

    # Create output directories
    raw_dir = output_dir / 'raw'
    red_dir = output_dir / 'reduced'
    mask_dir = output_dir / 'masks'
    raw_dir.mkdir(parents=True, exist_ok=True)
    red_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if verbosity > 0:
        print(f"🔍 Image/Label directories: {images_dir}, {labels_dir}")
    raw_paths = []
    # 0) Cut down to max_images if needed
    if max_images is not None:
        images = list(images_dir.iterdir())
        images = images[:max_images]
        if verbosity > 0:
            print(f"🔍 Limiting to {max_images} image-label pairs")
    else:
        images = list(images_dir.iterdir())

    # 1) Process each UUID folder: image + mask
    for uid_folder in images:
        if not uid_folder.is_dir():
            continue
        uid = uid_folder.name
        # Image file
        image = list(uid_folder.glob('*.czi'))
        img_path = image[0]
        # Label folder with same UUID
        lbl_folder = labels_dir / uid
        masks = list(lbl_folder.glob('*.czi'))
        if len(masks) != 1 or len(image) != 1:
            print(f"Skipping {uid}: expected 1 image and 1 mask, found {len(image)} and {len(masks)}")
            continue
        mask_path = masks[0]

        # Load and save mask
        mask = load_mask(mask_path)
        mask_out = mask_dir / f"{uid}_mask.npy"
        np.save(mask_out, mask)

        # Load and extract features
        gray = load_gray(img_path)
        fmap = extract_raw_features(gray, vgg_input_size)
        raw_out = raw_dir / f"{uid}_raw128.npy"
        np.save(raw_out, fmap)
        raw_paths.append(raw_out)

        if verbosity > 0:
            print(f"Processed {uid}: image shape={gray.shape}, mask shape={mask.shape}, \nfeatures shape={fmap.shape}")

    # 2) Fit projection on pooled features
    all_feats = []
    for p in raw_paths:
        arr = np.load(p)
        H, W, C = arr.shape
        all_feats.append(arr.reshape(-1, C))
    X = np.vstack(all_feats)
    print(f"Fitting projector on data shape {X.shape}")
    projector = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    projector.fit(X)
    transformer_path = models_dir / f"transformer_{n_components}.joblib"
    dump(projector, transformer_path)
    print(f"Saved transformer: {transformer_path}")

    # 3) Apply transformer to each raw feature map
    for p in raw_paths:
        arr = np.load(p)
        H, W, C = arr.shape
        flat = arr.reshape(-1, C)
        red = projector.transform(flat)
        fmap50 = red.reshape(H, W, n_components)
        red_out = red_dir / p.name.replace('_raw128', f'_feat{n_components}')
        np.save(red_out, fmap50.astype(np.float32))
        print(f"Saved reduced features: {red_out} shape={fmap50.shape}")

    print("Training data preparation complete.")
