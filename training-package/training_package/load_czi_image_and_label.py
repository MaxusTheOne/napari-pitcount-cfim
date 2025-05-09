import matplotlib.pyplot as plt
from czifile import CziFile
from pathlib import Path
import numpy as np


def czi_to_numpy(path, channel_index=0):
    print(f"Path: {path}")
    with CziFile(path) as czi:
        data = czi.asarray(order=0)

    print("[*convert] Raw shape:", data.shape)
    print(f"[*convert] Type: {data.dtype}")

    # For a typical CZI: (1, 1, 1, 1, Y, X, 1) → (Y, X)
    try:
        # Squeeze out unused dimensions
        squeezed = np.squeeze(data)
        # If shape is (C, Y, X), select the desired channel
        if squeezed.ndim == 3 and squeezed.shape[0] <= 4:
            img = squeezed[channel_index]
        elif squeezed.ndim == 2:
            img = squeezed
        else:
            raise ValueError(f"Unexpected shape after squeezing: {squeezed.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse CZI structure: {e}")

    return img

def load_and_preprocess(plot=True):
    # File paths
    file_path = Path(__file__).parent
    image_path = file_path / "training_data" / "Images" / "1c450669-fe0f-43f7-8e58-e4c74f89cb74" / "Tube 70 4 fluor.czi"
    label_path = file_path / "training_data" / "Labels" / "1c450669-fe0f-43f7-8e58-e4c74f89cb74" / "Tube 70 4 fluor.czi"

    # Load grayscale image
    image = czi_to_numpy(image_path)
    label = czi_to_numpy(label_path, channel_index=1)

    print(f"[*Load] Image loaded | Shape: {image.shape}")
    print(f"[*Load] Label loaded | Shape: {label.shape} | {label.ndim} | {label.dtype}")

    if plot:
        # Visual check
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap='gray', vmin=0, vmax=1)
        plt.title("Binary Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    np.save(file_path / "training_data" / "label.npy", label)
    return True

if __name__ == "__main__":
    load_and_preprocess()