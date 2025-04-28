from PyQt5.QtWidgets import QGroupBox
from qtpy.QtWidgets import QWidget, QVBoxLayout

from napari_pitcount_cfim.config.settings_handler import SettingsHandler
from napari_pitcount_cfim.image_handling.image_handler import ImageHandler
from napari_pitcount_cfim.result_handling.result_handler import ResultHandler


class MainWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self)
        self.image_handler = ImageHandler(parent=self, napari_viewer=self.viewer)
        self.result_handler = ResultHandler(parent=self)

        layout = QVBoxLayout()
        self.setLayout(layout)

        open_settings_group = self.setting_handler.init_ui()
        self.layout().addWidget(open_settings_group)
        pane = QGroupBox(self)
        pane.setTitle("Input / Output")
        pane.setLayout(QVBoxLayout())
        pane.layout().addWidget(self.image_handler.init_load_button_ui())
        pane.layout().addWidget(self.result_handler.init_output_button_ui())
        self.layout().addWidget(pane)


        self._update_widget_settings()


    def _update_widget_settings(self):
        """
        Update the settings of the widget.
        """
        settings = self.setting_handler.get_updated_settings()

        self.image_handler.set_output_path(settings.get("input_folder"))
        self.result_handler.set_output_path(settings.get("output_folder"))