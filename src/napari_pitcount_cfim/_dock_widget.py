from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit

from napari_pitcount_cfim.config.settings_handler import SettingsHandler


class MainWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self)

        layout = QVBoxLayout()
        self.setLayout(layout)

        hello_text_field = QLineEdit("Hello, World!")

        self.layout().addWidget(hello_text_field)

        open_settings_group = self.setting_handler.init_ui()
        self.layout().addWidget(open_settings_group)

