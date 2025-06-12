import json

import numpy as np
import joblib

from pathlib import Path

from napari_pitcount_cfim.pitcounter.transformers import czi_to_fmap, npy_to_fmap


class ModelUser:
    """
        A class for using a given model for pit-segmentation.
    """

    def __init__(self, model_folder: str | Path, prediction_settings: dict | None = None):
        self.model_file_dir: Path | None = None
        self.model_file: Path | None = None
        self.transformer_file: Path | None = None
        self.resize_to: int | None = None
        self.meta: dict = {}
        self.settings: dict = prediction_settings or {}

        self.transformer = None
        self.classifier_model = None


        self.set_from_folder(model_folder)

    def predict_from_npy(self, image_array, resize_to=None):
        if resize_to is None:
            resize_to = self.resize_to
        feat_map = npy_to_fmap(image_array, size=resize_to)
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
        classifier_model = self.classifier_model

        # TODO: Unify flattening
        H, W, C = feat_map.shape
        X = feat_map.reshape(-1, C)
        X = transformer.transform(X)

        # Predict
        y_pred = classifier_model.predict(X)

        # Reshape prediction back to image shape
        mask = y_pred.reshape(H, W).astype(np.uint8)

        if output_path:
            save_mask(mask, output_path)
        return mask

    def set_from_folder(self, model_folder: str | Path):
        """
            Set model_file, transformer_file, and meta on self from the given folder.
        """
        folder = Path(model_folder)
        self.model_file_dir = folder
        self.model_file, self.transformer_file, self.meta = self._unpack_model_folder(folder)

        resize = self.meta.get("resize_to", [])
        if not (isinstance(resize, (list, tuple)) and len(resize) == 2):
            raise ValueError(f"Invalid resize_to in metadata: {resize}")
        self.resize_to: tuple[int, int] = (int(resize[0]), int(resize[1]))

        self._set_load_models()

    def _set_load_models(self):
        if self.model_file.exists() and self.transformer_file.exists():
            self.classifier_model = joblib.load(self.model_file)
            self.transformer = joblib.load(self.transformer_file)
        else:
            raise FileNotFoundError(f"Model / Transformer not found, got: {self.model_file}, {self.transformer_file}")

    def _unpack_model_folder(self, model_folder: str | Path) -> tuple[Path, Path, dict]:
        """
            Identify and load model file, transformer file, and metadata from a folder.
            Returns (model_file, transformer_file, meta_dict).
        """
        folder = Path(model_folder)
        # Locate metadata JSON
        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            jsons = list(folder.glob("*.json"))
            if len(jsons) == 1:
                meta_path = jsons[0]
            else:
                raise FileNotFoundError(f"Cannot identify metadata JSON in {folder}")
        meta = json.loads(meta_path.read_text())

        # Locate model file
        model_file = folder / "model.joblib"
        if not model_file.exists() and meta.get("model_name"):
            candidate = folder / f"{meta['model_name']}.joblib"
            if candidate.exists():
                model_file = candidate

        # Locate transformer file
        transformer_file = folder / "transformer.joblib"
        if not transformer_file.exists() and meta.get("transformer_name"):
            candidate = folder / f"{meta['transformer_name']}.joblib"
            if candidate.exists():
                transformer_file = candidate

        # Fallback: assign leftover joblib files
        if not model_file.exists() or not transformer_file.exists():
            all_jobs = list(folder.glob("*.joblib"))
            # Exclude known
            remaining = [p for p in all_jobs if p not in (model_file, transformer_file)]
            # If one left and model missing
            if not model_file.exists() and len(remaining) == 1:
                model_file = remaining[0]
                remaining = []
            # If one left and transformer missing
            if not transformer_file.exists() and len(remaining) == 1:
                transformer_file = remaining[0]
        return model_file, transformer_file, meta



def save_mask(mask, output_path):
    from imageio import imwrite
    imwrite(output_path, mask * 255)
    print(f"✅ Mask saved to {output_path}")


