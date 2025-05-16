import json

import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from training_package.transformers import czi_to_fmap, npy_to_fmap
import argparse

class ModelUser:
    def __init__(self, model_dir: Path, transformer_file=None, meta_path=None):
        self.model_file_dir = model_dir.parent
        self.model_name = model_dir.stem
        self.model_file = self.model_file_dir / (self.model_name + ".joblib")
        self.transformer_file = transformer_file or (self.model_file_dir / "transformer.joblib")
        self.clf = joblib.load(self.model_file)
        self.transformer = joblib.load(self.transformer_file)

        if not meta_path:
            meta_path = self.model_file_dir / (self.model_name + "_metadata.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
            print(f"Meta: {self.meta}")

        self.resize_to = tuple(self.meta["resize_to"])

    def predict_from_npy(self, image_array, resize_to=None):
        if resize_to is None:
            resize_to = self.resize_to
        feat_map = npy_to_fmap(image_array, size=resize_to)
        print(f"I scream")
        mask = self.predict_mask(feat_map)
        return mask

    def predict_from_czi(self, czi_path, output_path, resize_to=None):
        if resize_to is None:
            resize_to = self.resize_to
        feat_map = czi_to_fmap(czi_path, size=resize_to)
        mask = self.predict_mask(feat_map, output_path)
        return mask


    def predict_mask(self, feat_map, output_path=None):
        transformer = self.transformer
        clf = self.clf

        H, W, C = feat_map.shape
        # Flatten and transform features
        X = feat_map.reshape(-1, C)
        X = transformer.transform(X)

        # Predict
        y_pred = clf.predict(X)

        # Reshape prediction back to image shape
        mask = y_pred.reshape(H, W).astype(np.uint8)

        if output_path:
            save_mask(mask, output_path)
        return mask


def save_mask(mask, output_path):
    # Optionally save
    from imageio import imwrite
    imwrite(output_path, mask * 255)
    print(f"✅ Mask saved to {output_path}")



