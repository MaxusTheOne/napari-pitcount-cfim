from pathlib import Path

import napari.layers
from qtpy.QtWidgets import QWidget, QPushButton, QFileDialog


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

    def get_all_labels(self):
        """
            Get all labels from the napari viewer.
        """
        if not self.viewer.layers:
            raise ValueError("No layers in the viewer.")
        return [layer.data for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]

    def init_load_button_ui(self):
        """
            Initialize the load button UI.
        """
        self.load_button = QPushButton("Load images from folder")
        self.load_button.clicked.connect(self._load_images)
        return self.load_button

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
        self.settings_handler.update_setting("file_settings.input_folder", folder)
        self.settings = self.settings_handler.get_updated_settings().get("file_settings")
        return True

    def _load_images(self):
        """
        Load images from a folder into the napari viewer.
        """
        # 0) Update the settings:
        self.settings = self.settings_handler.get_updated_settings().get("file_settings")
        # 1) If prompting is enabled, ask the user now:
        if self.settings.get("folder_prompt"):
            if not self._select_folder():
                return  # user cancelled, so do nothing

        # 2) Make sure we have a path (either from the dialog or pre‐set):
        if not self.settings.get("input_folder"):
            raise ValueError("input path is not set. Please set the input path before loading images.")

        # 3) Collect and open:
        img_dir   = Path(self.settings.get("input_folder"))
        img_paths = sorted(img_dir.iterdir())
        self.viewer.open(img_paths, plugin=None)
