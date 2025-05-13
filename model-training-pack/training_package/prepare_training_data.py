from pathlib import Path
import numpy as np
from czifile import CziFile
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T

# --- Paths ---
DATA_DIR = Path(__file__).parent / "training_data"
IMAGE_DIR = DATA_DIR / "Images"
LABEL_DIR = DATA_DIR / "Labels"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG = {
    "resize_to": None,
    "channel_index": 0,
    "max_images": None,
    "skip_existing": True,
    "verbosity": 2,
    "output_dir": Path(__file__).parent / "training_data" / "processed",
    "dry_run": False,
}

# --- VGG setup once ---
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:9]
vgg19.eval()


def find_max_even_size(image_dir, max_images=None, channel_index=0):
    min_h, min_w = float('inf'), float('inf')
    uuids = [d.name for d in image_dir.iterdir() if d.is_dir()]
    if max_images:
        uuids = uuids[:max_images]
    for uid in uuids:
        path = image_dir / uid / "*.czi"
        files = list(image_dir.glob(f"{uid}/*.czi"))
        if not files:
            continue
        img = load_czi(files[0], channel_index=channel_index)
        h, w = img.shape[:2]
        min_h, min_w = min(min_h, h), min(min_w, w)
    # round down to even
    return int(min_h) - int(min_h) % 2, int(min_w) - int(min_w) % 2

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

    resized = cv2.resize(image_array, dsize=resize_to)
    image_rgb = cv2.cvtColor((resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0)
    if CONFIG["verbosity"] > 2:
        print(f"🔍 Extracting features from {image_array.shape} to {resize_to}")
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
def process_all(config=None):
    if config is None:
        config = CONFIG
    else:
        config = {**CONFIG, **config}

    image_dir = config["input_dir"] / "Images"
    label_dir = config["input_dir"] / "Labels"
    if config.get("resize_to") is None:
        config["resize_to"] = find_max_even_size(
            image_dir,
            max_images=config["max_images"],
            channel_index=config["channel_index"]
        )
        if config["verbosity"] > 0:
            print(f"🔍 Found max even size: {config['resize_to']}")

    if config["verbosity"] > 1:
        print(f"Processing settings: {config}")
    uuids = [d.name for d in image_dir.iterdir() if d.is_dir()]
    if config["max_images"] is not None:
        uuids = uuids[:config["max_images"]]

    if config["verbosity"] > 0:
        if len(uuids) == config["max_images"]:
            print(f"🔍 Found {len(uuids)} image-label pairs (limited to {config['max_images']})")
        else: print(f"🔍 Found {len(uuids)} image-label pairs")

        print(f'output_dir: {config["output_dir"]}')


    for uid in uuids:
        out_dir = config["output_dir"] / uid
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_path = out_dir / "features.npy"
        label_path = out_dir / "label.npy"

        if config["skip_existing"] and feature_path.exists() and label_path.exists():
            if config["verbosity"] > 0:
                print(f"❕ Skipping {uid}, already processed.")
            continue

        try:
            image_files = list((image_dir / uid).glob("*.czi"))
            label_files = list((label_dir / uid).glob("*.czi"))

            if len(image_files) != 1 or len(label_files) != 1:
                # Remove the uuid directory if it has no valid files
                out_dir.rmdir()
                raise RuntimeError(f"{uid}: expected 1 image and 1 label, found {len(image_files)}, {len(label_files)}")

            image = load_czi(image_files[0])
            label = load_czi(label_files[0], channel_index=config["channel_index"])

            if not np.any(label != 0):
                raise ValueError("Label array contains only zeros")

            # ninja code: label > 1 makes a bool array
            label[label > 1] = 1

            features = extract_vgg_features(image, config["resize_to"])

            if not config["dry_run"]:
                np.save(feature_path, features)
                np.save(label_path, label)

            if config["verbosity"] > 0:
                print(f"✅ Processed {uid} | Features: {features.shape} | Label: {label.shape}")

        except RuntimeError as e:
            print(f"❌ Error with {uid}: {e}")
        except ValueError as e:
            print(f"❌❌❌ Error with {uid}: {e}")

    return config


if __name__ == "__main__":
    process_all()

