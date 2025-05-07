import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load from any one processed pair
default_UUID = "9de436ac-5f1c-4b35-9a45-913d2dd25c68"  # Replace with an actual folder name
def preview_pair(uuid=default_UUID):
    PAIR_DIR = Path(__file__).parent / "data" / "processed" / uuid

    feature_path = PAIR_DIR / "features.npy"
    label_path = PAIR_DIR / "label.npy"

    # Load
    features = np.load(feature_path)
    label = np.load(label_path)

    print(f"[*Load] Image loaded | Shape: {features.shape}")
    print(f"[*Load] Label loaded | Shape: {label.shape} | {label.ndim} | {label.dtype}")

    # Extract grayscale image approximation (e.g. channel 0 of deep features)
    gray_image = features[..., 0]

    # Display
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"Approx. Grayscale Image\nUUID: {uuid}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray', vmin=0, vmax=1)
    plt.title("Binary Label Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preview_pair("1c450669-fe0f-43f7-8e58-e4c74f89cb74")