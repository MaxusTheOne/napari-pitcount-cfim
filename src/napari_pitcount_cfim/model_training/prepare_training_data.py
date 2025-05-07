from pathlib import Path
import numpy as np
from czifile import CziFile
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import os

# --- Paths ---
DATA_DIR = Path(__file__).parent / "data"
IMAGE_DIR = DATA_DIR / "Images"
LABEL_DIR = DATA_DIR / "Labels"
PROCESSED_DIR = DATA_DIR / "processed"
RESIZE_TO = (256, 256)  # Can adjust later if needed

# --- VGG setup once ---
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:9]
vgg19.eval()

# --- Helper: load .czi image ---
def load_czi(path, channel_index=0):
    with CziFile(path) as czi:
        data = czi.asarray()
    squeezed = np.squeeze(data)
    if squeezed.ndim == 3:
        return squeezed[channel_index].astype(np.uint16)
    return squeezed.astype(np.uint16)

# --- Helper: extract features from grayscale uint16 image ---
def extract_vgg_features(image_array, resize_to):
    image_array = image_array.astype(np.float32)
    image_array /= image_array.max()  # scale to [0, 1]

    resized = cv2.resize(image_array, resize_to)
    image_rgb = cv2.cvtColor((resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        features = vgg19(input_tensor).squeeze(0)
        upsampled = torch.nn.functional.interpolate(
            features.unsqueeze(0),
            size=image_array.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    return upsampled.permute(1, 2, 0).numpy()  # (H, W, C)

# --- Main loop ---
def process_all():
    uuids = [d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(uuids)} image-label pairs")

    for uid in uuids:
        out_dir = PROCESSED_DIR / uid
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_path = out_dir / "features.npy"
        label_path = out_dir / "label.npy"

        if feature_path.exists() and label_path.exists():
            print(f"✅ Skipping {uid}, already processed.")
            continue

        try:
            image_grab = list((IMAGE_DIR / uid).glob("*.czi"))
            label_grab = list((LABEL_DIR / uid).glob("*.czi"))

            if len(image_grab) != 1 or len(label_grab) != 1:
                raise RuntimeError(
                    f"Expected one .czi in each folder for {uid}, found {len(image_grab)} image(s), {len(label_grab)} label(s)")

            image_czi = image_grab[0]
            label_czi = label_grab[0]

            image = load_czi(image_czi)
            label = load_czi(label_czi, channel_index=1)

            features = extract_vgg_features(image, RESIZE_TO)

            np.save(feature_path, features)
            np.save(label_path, label)

            print(f"✅ Saved: {uid} | Features: {features.shape} | Label: {label.shape}")
        except Exception as e:
            print(f"❌ Failed on {uid}: {e}")

if __name__ == "__main__":
    process_all()
