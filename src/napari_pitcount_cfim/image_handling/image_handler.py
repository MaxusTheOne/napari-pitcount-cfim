from pathlib import Path

from PyQt5.QtWidgets import QWidget, QPushButton


class ImageHandler(QWidget):
    """
        A class to handle and manage transfer of images between the napari viewer and the plugin.
    """
    def __init__(self, napari_viewer, parent=None, output_path=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.parent = parent
        self.output_path = output_path
        self.load_button = None

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
        self.output_path = path

    def _load_images(self):
        """
            Load images from a folder into the napari viewer.
        """
        if self.output_path is None:
            raise ValueError("Output path is not set. Please set the output path before loading images.")

        img_dir = Path(self.output_path)
        img_paths = sorted(img_dir.iterdir())

        # 2. Create a Viewer and open them all
        # plugin=None tells napari to infer which plugin to use for each path
        self.viewer.open(img_paths, plugin=None)
