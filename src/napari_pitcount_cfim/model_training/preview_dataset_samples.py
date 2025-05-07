import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
uuids = sorted([p.name for p in PROCESSED_DIR.iterdir() if p.is_dir()])

def show_preview(uuid):
    pair_path = PROCESSED_DIR / uuid
    features = np.load(pair_path / "features.npy")
    label = np.load(pair_path / "label.npy")

    # Approx grayscale from first VGG channel
    image = features[..., 0]

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"UUID: {uuid}", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Approx. Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap="gray", vmin=0, vmax=1)
    plt.title("Binary Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Show first 5 pairs
for i, uuid in enumerate(uuids[:5]):
    print(f"🔍 Previewing {uuid} ({i+1}/5)")
    show_preview(uuid)
