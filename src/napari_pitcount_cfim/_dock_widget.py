<<<<<<< Updated upstream
from PyQt5.QtWidgets import QGroupBox
from qtpy.QtWidgets import QWidget, QVBoxLayout
=======
import pathlib

from PyQt5.QtWidgets import QLabel
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLayout
>>>>>>> Stashed changes

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
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(layout)

        self._add_logo()

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
<<<<<<< Updated upstream
        self.result_handler.set_output_path(settings.get("output_folder"))
=======

    def _add_logo(self):
        """
        Add the logo to the widget.
        """
        path = pathlib.Path(__file__).parent / "logo" / "CFIM_logo_small.png"
        logo_label = QLabel()
        logo_label.setText(f"<img src='{path}' width='320'/>")
        print(f"Dev | Logo path: {path}")
        self.layout().addWidget(logo_label)
>>>>>>> Stashed changes
