from pathlib import Path

import napari.layers
import numpy as np
from qtpy.QtWidgets import QWidget, QPushButton, QFileDialog

# Default values
ACCEPTABLE_SYNONYMS = {
    "uuid": "unique_id",
    "meta": "metadata",

}

class ImageHandler(QWidget):
    """
        A class to handle and manage transfer of images between the napari viewer and the plugin.
    """
    def __init__(self, napari_viewer, parent=None, settings_handler=None):
        super().__init__(parent)
        self.settings_handler = settings_handler

        if settings_handler is None:
            raise ValueError("Settings handler is not set. Please provide a settings handler.")

        self.settings = settings_handler.get_settings().get("file_settings")
        self.viewer = napari_viewer
        self.parent = parent
        self.input_path = self.settings.get("input_folder")
        self.load_button = None

    def get_all_images(self):
        """
            Get all images from the napari viewer.
        """
        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")
        return [layer.data for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

    def get_all_images_with_names(self):
        """
            Get all images from the napari viewer with their names.
        """
        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")
        return [(layer.name, layer.data) for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

    def get_all_images_props(self, props=None):
        """
            Get all images from the napari viewer with their properties.
        """
        if props is None:
            props = ["data", "name", "uuid"]

        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")

        if self.settings_handler.get_settings()["debug_settings"].get("debug"):
            print(f"Debug | Getting all images with properties: {props}")

            # Change synonyms to correct property names
            props = [ACCEPTABLE_SYNONYMS.get(prop, prop) for prop in props]

        return [
            {prop: getattr(layer, prop) for prop in props} for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)
        ]

    def get_all_labels(self):
        """
            Get all labels from the napari viewer.
        """
        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")
        return [layer.data for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]

    def add_image(self, image, name=None):
        """
            Add an image to the napari viewer.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array.")
        if name is None:
            name = f"Image {len(self.viewer.layers)}"
        self.viewer.add_image(image, name=name)

    def add_label(self, label, name=None, scale=None, metadata=None):
        """
            Add a label to the napari viewer.
        """

        if not isinstance(label, np.ndarray):
            raise TypeError("Label must be a numpy array.")
        if scale is None:
            scale = self.get_scale(0)
        if name is None:
            name = f"Label {len(self.viewer.layers)}"
        self.viewer.add_labels(label, name=name, scale=scale, blending="additive", metadata=metadata)

    def init_load_button_ui(self):
        """
            Initialize the load button UI.
        """
        self.load_button = QPushButton("Load images from folder")
        self.load_button.clicked.connect(self._load_images)
        return self.load_button

    def load_images(self, load_settings):
        """
            Load images from a folder into the napari viewer.
            If path is provided, it will be used as the input path.
        """
        path = load_settings.get("input_folder", "")
        verbosity = load_settings.get("verbosity", 0)
        self._load_images(path, verbosity)

    def set_output_path(self, path):
        """
            Set the output path for the images.
        """
        if not isinstance(path, str):
            raise ValueError("Output path must be a string.")
        self.input_path = path


    def get_scale(self, index):
        """
        Get the scale of the image at the given index.
        """
        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")
        if index >= len(self.viewer.layers):
            raise IndexError("Index out of range.")
        layer = self.viewer.layers[index]
        if not isinstance(layer, napari.layers.Image):
            raise TypeError("Layer is not an image.")
        return layer.scale

    def _select_folder(self) -> bool:
        """
        Pop up a folder‐selection dialog and store the result in self.output_path.
        Returns False if the user cancels.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select image folder",
            str(self.input_path)
        )
        if not folder:
            return False
        self.settings_handler.update_settings("file_settings.input_folder", folder)
        self.settings = self.settings_handler.get_updated_settings().get("file_settings")
        return True

    def _load_images(self, path: Path=None, verbosity: int=0):
        """
        Load images from a folder into the napari viewer.
        """
        # 0) Update the settings:
        self.settings = self.settings_handler.get_updated_settings().get("file_settings")

        if path:
            if path.exists() and path.is_dir():
                folder_path = Path(path)
                if verbosity >= 1:
                    print(f"Loading images from given path: {folder_path}")
            else:
                if self.settings.get("input_folder"):
                    folder_path = Path(self.settings.get("input_folder"))
                    if verbosity >= 1:
                        print(f"No path given, using input_path from settings: {folder_path}")
                else:
                    raise ValueError(f"Expected a valid path, but got: {path}. Please provide a valid path or set the input folder in settings.")

        else:
            if self.settings.get("folder_prompt"):
                if not self._select_folder():
                    return  # user cancelled, so do nothing

            # 2) Make sure we have a path (either from the dialog or pre‐set):
            if not self.settings.get("input_folder"):
                raise ValueError("input path is not set. Please set the input path before loading images.")
            else:
                folder_path = self.settings.get("input_folder")


        img_dir   = Path(folder_path)
        img_paths = sorted(img_dir.iterdir())
        self.viewer.open(img_paths, plugin=None)
