from pathlib import Path

from PyQt5.QtWidgets import QWidget, QPushButton, QFileDialog


class ImageHandler(QWidget):
    """
        A class to handle and manage transfer of images between the napari viewer and the plugin.
    """
    def __init__(self, napari_viewer, parent=None, output_path=None, prompt_for_folder=True):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.parent = parent
        self.output_path = output_path
        self.prompt_for_folder = prompt_for_folder
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

    def enable_folder_prompt(self, enabled: bool):
        """
        Turn on/off the folder‐selection dialog.
        """
        self.prompt_for_folder = bool(enabled)

    def _select_folder(self) -> bool:
        """
        Pop up a folder‐selection dialog and store the result in self.output_path.
        Returns False if the user cancels.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select image folder",
            str(self.output_path)
        )
        if not folder:
            return False
        self.output_path = folder
        return True

    def _load_images(self):
        """
        Load images from a folder into the napari viewer.
        """
        # 1) If prompting is enabled, ask the user now:
        if self.prompt_for_folder:
            if not self._select_folder():
                return  # user cancelled, so do nothing

        # 2) Make sure we have a path (either from the dialog or pre‐set):
        if not self.output_path:
            raise ValueError("Output path is not set. Please set the output path before loading images.")

        # 3) Collect and open:
        img_dir   = Path(self.output_path)
        img_paths = sorted(img_dir.iterdir())
        self.viewer.open(img_paths, plugin=None)
