import cv2
import numpy as np
import czifile
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from torchvision.models import vgg19, VGG19_Weights

# 1) Prepare VGG19 up to conv2_2 (first 9 layers)
_vgg_conv2_2 = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
_vgg_conv2_2.eval()

# 2) ImageNet normalization for VGG inputs
_VGG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

def czi_to_fmap(
    czi_path: Path | str,
    size: tuple[int,int] = None
) -> np.ndarray:
    """
    Load a .czi microscopy image and return its raw VGG conv2_2 feature map.

    Args:
      czi_path: Path to your .czi file.
      size: (height, width) to resize for VGG before extracting features.

    Returns:
      fmap: numpy array of shape (H, W, 128), where H/W are the original image dims.
    """
    if size is None:
        print(f"Got None for size, using default (256, 256)")
        size = (256, 256)
    # --- 1. Load & squeeze to 2D or 3D array ---
    arr = czifile.imread(str(czi_path))
    arr = np.squeeze(arr)  # e.g. (1,1,C,H,W,1) → (C,H,W) or (H,W,C) or (H,W)

    # --- 2. Convert to single‐channel grayscale uint8 ---
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        # if shape is (C,H,W), transpose to (H,W,C)
        if arr.shape[0] in (1,3,4):
            arr = arr.transpose(1,2,0)
        # now arr is (H,W,channels)
        chans = arr.shape[2]
        if chans >= 3:
            # take first 3 channels as RGB
            rgb = arr[..., :3].astype(np.float32)
            if rgb.max() > 0:
                rgb = rgb / rgb.max() * 255
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            # single‐channel fallback
            gray = arr[...,0]
    else:
        raise ValueError(f"Unexpected arr.ndim={arr.ndim} for {czi_path}")

    # ensure uint8 in [0,255]
    gray = gray.astype(np.float32)
    if gray.max() > 0:
        gray = gray / gray.max() * 255
    gray = gray.astype(np.uint8)

    H0, W0 = gray.shape

    # --- 3. Resize for VGG19 input ---
    resized = cv2.resize(gray, dsize=size[::-1], interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # --- 4. Extract conv2_2 features and upsample back ---
    x = _VGG_TRANSFORM(rgb).unsqueeze(0)            # 1×3×h×w
    with torch.no_grad():
        feats = _vgg_conv2_2(x)                     # 1×128×h'×w'
        feats = F.interpolate(feats, size=size,
                              mode='bilinear',
                              align_corners=False)
    fmap = feats.squeeze(0).permute(1,2,0).cpu().numpy()  # h×w×128

    # --- 5. Upsample feature‐map to original image size if needed ---
    if size != (H0, W0):
        fmap = cv2.resize(fmap, (W0, H0), interpolation=cv2.INTER_LINEAR)
        fmap = fmap.reshape(H0, W0, -1)

    return fmap.astype(np.float32), arr

def npy_to_fmap(
    npy_image: np.ndarray,
    size: tuple[int,int] = None
) -> np.ndarray:
    """
    Load a .npy microscopy image and return its raw VGG conv2_2 feature map.

    Args:
      npy_image: npy array.
      size: (height, width) to resize for VGG before extracting features.

    Returns:
      fmap: numpy array of shape (H, W, 128), where H/W are the original image dims.
    """
    if size is None:
        print(f"Got None for size, using default (256, 256)")
        size = (256, 256)

    # --- 1. Load & squeeze to 2D or 3D array ---
    arr = npy_image
    arr = np.squeeze(arr)

    # --- 2. Convert to single‐channel grayscale uint8 ---
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        # if shape is (C,H,W), transpose to (H,W,C)
        if arr.shape[0] in (1,3,4):
            arr = arr.transpose(1,2,0)
        # now arr is (H,W,channels)
        chans = arr.shape[2]
        if chans >= 3:
            # take first 3 channels as RGB
            rgb = arr[..., :3].astype(np.float32)
            if rgb.max() > 0:
                rgb = rgb / rgb.max() * 255
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            # single‐channel fallback
            gray = arr[...,0]
    else:
        raise ValueError(f"Unexpected arr.ndim={arr.ndim} for {npy_image}")

    # ensure uint8 in [0,255]
    gray = gray.astype(np.float32)
    if gray.max() > 0:
        gray = gray / gray.max() * 255
    gray = gray.astype(np.uint8)

    H0, W0 = gray.shape

    # --- 3. Resize for VGG19 input ---
    resized = cv2.resize(gray, dsize=size[::-1], interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # --- 4. Extract conv2_2 features and upsample back ---
    x = _VGG_TRANSFORM(rgb).unsqueeze(0)            # 1×3×h×w
    with torch.no_grad():
        feats = _vgg_conv2_2(x)                     # 1×128×h'×w'
        feats = F.interpolate(feats, size=size,
                               mode='bilinear',
                               align_corners=False)
    fmap = feats.squeeze(0).permute(1,2,0).cpu().numpy()  # h×w×128

    # --- 5. Upsample feature‐map to original image size if needed ---
    if size != (H0, W0):
        fmap = cv2.resize(fmap, (W0, H0), interpolation=cv2.INTER_LINEAR)
        fmap = fmap.reshape(H0, W0, -1)

    return fmap.astype(np.float32)
