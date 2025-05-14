#!/usr/bin/env python3
"""
prepare_training_data.py

Extract deep features (128-dim) from grayscale microscopy images using VGG19 up to conv2_2,
fit a 50-d Gaussian random projection on the pooled features, and save both raw and reduced features
for downstream classifier training.

This module exposes a `prepare_training_data` function that can be imported and called by other scripts.
It also supports command-line execution with a JSON config or explicit arguments.
"""
import os
import json
import argparse
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

# Define VGG preprocessing and model (up to conv2_2)
VGG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
vgg_conv2_2 = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
vgg_conv2_2.eval()


def load_gray(path: Path) -> np.ndarray:
    """
    Load an image (CZI or standard formats) as 2D uint8 grayscale.
    """
    arr = czifile.imread(str(path)) if path.suffix.lower() == '.czi' else cv2.imread(
        str(path), cv2.IMREAD_UNCHANGED
    )
    arr = np.squeeze(arr)
    # Convert multi-channel to gray
    if arr.ndim == 3:
        channels = arr.shape[-1]
        if channels in (3, 4):
            rgb = arr[..., :3] if channels == 4 else arr
            rgb = rgb.astype(np.float32)
            if rgb.max() > 0:
                rgb = rgb / rgb.max() * 255
            arr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[..., 0]
    # Scale to uint8
    arr = arr.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    return arr.astype(np.uint8)


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
        feats = vgg_conv2_2(x)  # 1×128×h'×w'
        feats = F.interpolate(feats, size=vgg_input_size, mode='bilinear', align_corners=False)
    fmap = feats.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Upsample to original resolution if different
    if vgg_input_size != (H0, W0):
        fmap = cv2.resize(
            fmap, (W0, H0), interpolation=cv2.INTER_LINEAR
        ).reshape(H0, W0, -1)
    return fmap.astype(np.float32)


def prepare_training_data(
    images_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    n_components: int = 50,
    random_state: int = 42,
    vgg_input_size: tuple = (256, 256)
):
    """
    Extract raw features, fit a random projection, and save reduced features.

    images_dir: Path to UUID-folders with images
    processed_dir: Path under which 'raw/' and 'reduced/' will be created
    models_dir: Path where transformer_50.joblib will be saved
    n_components: Number of projected dimensions
    random_state: Seed for reproducibility
    vgg_input_size: Input size for VGG feature extraction
    """
    raw_dir = processed_dir / 'raw'
    red_dir = processed_dir / 'reduced'
    raw_dir.mkdir(parents=True, exist_ok=True)
    red_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract raw features
    raw_paths = []
    for uid_folder in images_dir.iterdir():
        if not uid_folder.is_dir():
            continue
        imgs = list(uid_folder.glob('*'))
        if len(imgs) != 1:
            print(f"Skipping {uid_folder.name}: expected 1 image, found {len(imgs)}")
            continue
        gray = load_gray(imgs[0])
        fmap = extract_raw_features(gray, vgg_input_size)
        out_raw = raw_dir / f"{uid_folder.name}_raw128.npy"
        np.save(out_raw, fmap)
        raw_paths.append(out_raw)
        print(f"Saved raw: {out_raw} shape={fmap.shape}")

    # Step 2: fit projector on all raw features
    all_feats = []
    for p in raw_paths:
        arr = np.load(p)
        H, W, C = arr.shape
        all_feats.append(arr.reshape(-1, C))
    X = np.vstack(all_feats)
    print(f"Fitting projector on data shape {X.shape}")
    projector = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state
    )
    projector.fit(X)
    trans_path = models_dir / f"transformer_{n_components}.joblib"
    dump(projector, trans_path)
    print(f"Saved transformer: {trans_path}")

    # Step 3: apply transformer per image
    for p in raw_paths:
        arr = np.load(p)
        H, W, C = arr.shape
        flat = arr.reshape(-1, C)
        red = projector.transform(flat)
        fmap50 = red.reshape(H, W, n_components)
        out_red = red_dir / p.name.replace('_raw128', f'_feat{n_components}')
        np.save(out_red, fmap50.astype(np.float32))
        print(f"Saved reduced: {out_red} shape={fmap50.shape}")

    print("Finished preparing training data.")
