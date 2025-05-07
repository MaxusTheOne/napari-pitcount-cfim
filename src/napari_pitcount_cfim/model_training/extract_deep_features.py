
from napari_pitcount_cfim.model_training.load_czi_image_and_label import czi_to_numpy
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import torch
import cv2



def extract_vgg_features(image_array, resize_to=(256, 256)):
    """
    Extract deep features from a grayscale image using VGG19 up to conv2_2.
    """
    # Normalize and convert image to 8-bit float for PyTorch
    image_array = image_array.astype(np.float32)
    image_array /= image_array.max()  # Scale to [0, 1]

    # Resize for speed
    image_resized = cv2.resize(image_array, resize_to)

    # Convert grayscale → RGB (required for VGG)
    image_rgb = cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Normalize for VGG
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    print(f"Resized shape: {image_rgb.shape} | {image_rgb.max()} | {image_rgb.dtype}")
    input_tensor = transform(image_rgb).unsqueeze(0)  # shape: (1, 3, H, W)

    # VGG19 up to conv2_2
    vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
    vgg19_model.eval()

    with torch.no_grad():
        features = vgg19_model(input_tensor).squeeze(0)  # shape: (C, h, w)
        upsampled = torch.nn.functional.interpolate(
            features.unsqueeze(0),
            size=image_array.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # shape: (C, H, W)

    return upsampled.permute(1, 2, 0).numpy()  # shape: (H, W, C)

# --- Example usage ---
if __name__ == "__main__":
    dir_path = Path(__file__).parent
    # Load image (not label) from verified .czi path
    image_path = dir_path / "data" / "Images" / "1c450669-fe0f-43f7-8e58-e4c74f89cb74" / "Tube 70 4 fluor.czi"
    np_image = czi_to_numpy(image_path)

    vgg_features = extract_vgg_features(np_image)

    print(f"✅ Deep features extracted. Shape: {vgg_features.shape}")  # (H, W, C), e.g. (1582, 1572, 128)

    # Save for training phase
    np.save(dir_path / "data" / "deep_features.npy", vgg_features)