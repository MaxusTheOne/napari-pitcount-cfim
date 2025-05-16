#!/usr/bin/env python3
"""
visualize_predictions.py

Load .czi (or image) files, use a trained model and transformer to predict binary masks,
and display overlays of image+prediction for each.

Usage:
    python visualize_predictions.py \
        --images-dir /path/to/Images \
        --model-dir  /path/to/model_folder

Where `model_dir` contains:
  - A classifier `.joblib` (e.g. `rf_model.joblib`)
  - A transformer `.joblib` (e.g. `transformer_50.joblib`)
  - Optionally, `<model_name>_metadata.json`

Visualizes each image with predicted mask overlay.
"""
import argparse
from pathlib import Path
import numpy as np
import czifile
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from joblib import load
import matplotlib.pyplot as plt

# --- Prepare VGG up to conv2_2 ---
vgg_conv2_2 = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
vgg_conv2_2.eval()
_VGG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def czi_to_fmap(czi_path: Path, vgg_input_size=(256,256)) -> np.ndarray:
    """
    Load .czi or image, convert to grayscale, extract raw conv2_2 features
    and upsample to original resolution. Returns H×W×128 array.
    """
    arr = czifile.imread(str(czi_path)) if czi_path.suffix.lower() == '.czi' else cv2.imread(
        str(czi_path), cv2.IMREAD_UNCHANGED)
    arr = np.squeeze(arr)
    # to 2D grayscale
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        # (C,H,W) -> (H,W,C)
        if arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
            arr = arr.transpose(1,2,0)
        chans = arr.shape[-1]
        if chans >= 3:
            rgb = arr[..., :3].astype(np.float32)
            if rgb.max()>0: rgb = rgb / rgb.max()*255
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = arr[...,0].astype(np.float32)
    else:
        raise ValueError(f"Cannot interpret image with shape {arr.shape}")
    # normalize
    gray = gray.astype(np.float32)
    if gray.max()>gray.min(): gray = (gray - gray.min())/(gray.max()-gray.min())
    gray = (gray*255).astype(np.uint8)
    H0,W0 = gray.shape
    # resize for VGG
    resized = cv2.resize(gray, dsize=(vgg_input_size[1], vgg_input_size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    # forward conv2_2
    x = _VGG_TRANSFORM(rgb).unsqueeze(0)
    with torch.no_grad():
        feats = vgg_conv2_2(x)
        feats = F.interpolate(feats, size=vgg_input_size, mode='bilinear', align_corners=False)
    fmap = feats.squeeze(0).permute(1,2,0).cpu().numpy()
    # upsample to original
    if (vgg_input_size[0], vgg_input_size[1]) != (H0,W0):
        fmap = cv2.resize(fmap, (W0,H0), interpolation=cv2.INTER_LINEAR)
        fmap = fmap.reshape(H0,W0,-1)
    return fmap.astype(np.float32)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(1,0,0), alpha=0.5) -> np.ndarray:
    # image: H×W float [0,1], mask: H×W binary
    H,W = image.shape
    base = np.stack([image]*3, -1)
    overlay = base.copy()
    for c in range(3): overlay[...,c][mask==1] = color[c]
    return overlay*alpha + base*(1-alpha)


def main(images_dir: str, model_dir: str, vgg_input_size=(256,256)):
    images_dir = Path(images_dir)
    model_dir = Path(model_dir)
    # load classifier
    cls_paths = list(model_dir.glob('*.joblib'))
    transformer_path = model_dir / ''
    model_path = None
    transformer = None
    for p in cls_paths:
        if 'transformer' in p.stem:
            transformer_path = p
        else:
            model_path = p
    if not model_path or not transformer_path:
        raise FileNotFoundError("Need both a model.joblib and transformer_*.joblib in model_dir")
    model = load(model_path)
    transformer = load(transformer_path)
    # process images
    for img_file in images_dir.rglob('*.czi'):
        uid = img_file.stem
        # 1) get fmap
        fmap = czi_to_fmap(img_file, vgg_input_size)
        H,W,_ = fmap.shape
        # 2) flatten & transform
        flat = fmap.reshape(-1, fmap.shape[-1])
        feat50 = transformer.transform(flat)
        # 3) predict and reshape
        pred_flat = model.predict(feat50)
        pred_map = pred_flat.reshape(H, W)
        # normalize image for display
        img_gray = cv2.cvtColor(cv2.resize(fmap[:,:,0], (W,H)), cv2.COLOR_GRAY2BGR) # or reload original
        # create overlay
        img_disp = overlay_mask((img_gray[...,0]/255), pred_map)
        # show
        plt.figure(figsize=(4,4))
        plt.title(f"{uid} - Prediction")
        plt.imshow(img_disp)
        plt.axis('off')
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir',"-id", required=True)
    parser.add_argument('--model-dir', "-md", required=True)
    parser.add_argument('--vgg-input-size', "-is", nargs=2, type=int, default=[1024,1024])
    args = parser.parse_args()
    main(args.images_dir, args.model_dir, tuple(args.vgg_input_size))
