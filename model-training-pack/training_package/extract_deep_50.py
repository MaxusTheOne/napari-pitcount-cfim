import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from joblib import load, dump
from sklearn.random_projection import GaussianRandomProjection
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F


class DeepFeatureExtractor50:
    """
    Extract 50-dimensional deep features using VGG19 up to conv2_2 and a random projection.
    """
    def __init__(self, transformer_path: Path = None):
        # Initialize VGG feature extractor
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
        self.vgg.eval()
        # Prepare image transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Determine transformer file path
        if transformer_path is None:
            transformer_path = Path(__file__).parent / "models" / "transformer_50.joblib"
        transformer_path.parent.mkdir(parents=True, exist_ok=True)
        # Load or create random projection transformer
        if transformer_path.exists():
            self.proj = load(transformer_path)
        else:
            logging.info(f"Transformer not found. Creating new one at: {transformer_path}")
            self.proj = GaussianRandomProjection(n_components=50, random_state=42)
        self.path = transformer_path
        self._fitted = False

    def fit(self, gray_images: List[np.ndarray], resize_to: tuple):
        # Extract VGG features for *all* gray_images, stack, then
        all_feats = []
        for img in gray_images:
            fmap = self._vgg_forward(img, resize_to)  # H×W×128
            H, W, C = fmap.shape
            all_feats.append(fmap.reshape(-1, C))
        X = np.vstack(all_feats)
        self.proj.fit(X)
        self._fitted = True

    def extract(self, gray_image: np.ndarray, resize_to: tuple) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Transformer not yet fit! Call .fit(...) first.")
        fmap = self._vgg_forward(gray_image, resize_to)
        H, W, C = fmap.shape
        flat = fmap.reshape(-1, C)
        red = self.proj.transform(flat)  # only transform!
        return red.reshape(H, W, -1).astype(np.float32)

    def save(self):
        dump(self.proj, self.path)

    def save(self):
        """
        Save the transformer to model folder.
        """
        dump(self.proj, self.path)
        logging.info(f"Transformer saved to {self.path}")

# For backward compatibility
# instantiate default extractor and function alias
_default_extractor = DeepFeatureExtractor50()
def extract_deep_features_50(gray_image: np.ndarray, resize_to: tuple, model_file_path: Path = None):
    return _default_extractor.extract_fit(gray_image, resize_to)

