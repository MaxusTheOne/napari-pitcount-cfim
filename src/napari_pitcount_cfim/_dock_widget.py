from qtpy.QtWidgets import QWidget, QVBoxLayout

from napari_pitcount_cfim.config.settings_handler import SettingsHandler
from napari_pitcount_cfim.image_handling.image_handler import ImageHandler


class MainWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self)
        self.image_handler = ImageHandler(parent=self, napari_viewer=self.viewer)

        layout = QVBoxLayout()
        self.setLayout(layout)

        open_settings_group = self.setting_handler.init_ui()
        self.layout().addWidget(open_settings_group)
        self.layout().addWidget(self.image_handler.init_load_button_ui())


        self._update_widget_settings()


    def _update_widget_settings(self):
        """
        Update the settings of the widget.
        """
        settings = self.setting_handler.get_updated_settings()

        self.image_handler.set_output_path(settings.get("input_folder"))