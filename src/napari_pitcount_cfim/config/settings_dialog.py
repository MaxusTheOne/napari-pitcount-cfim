from enum import Enum

from PyQt6.QtWidgets import QGroupBox
from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from qtpy.QtWidgets import QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, QDialogButtonBox, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit

from napari_pitcount_cfim.config.settings_structure import CFIMSettings


class SettingsDialog(QDialog):
    def __init__(self, initial_settings: CFIMSettings, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Plugin Settings")
        self.resize(500, 400)

        # Store the initial settings
        self.initial_settings = initial_settings
        self.settings: CFIMSettings

        # Create tabs for Basic and Advanced settings
        self.tabs = QTabWidget(self)
        self.basic_tab = QWidget(self)
        self.advanced_tab = QWidget(self)
        self.tabs.addTab(self.basic_tab, "Basic Settings")
        self.tabs.addTab(self.advanced_tab, "Advanced Settings")

        # Layouts for each tab
        self.basic_layout = QVBoxLayout(self.basic_tab)
        self.advanced_layout = QVBoxLayout(self.advanced_tab)

        # Create dialog buttons: Apply, Cancel, Done
        buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        buttons.button(QDialogButtonBox.Apply).setText("Apply")
        buttons.button(QDialogButtonBox.Ok).setText("Done")
        # Connect buttons to handlers
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)  # close without saving
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._save_settings)
        buttons.button(QDialogButtonBox.Ok).clicked.connect(self._save_and_close)

        # Assemble layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(buttons)

        # Populate the tabs with settings fields
        self._add_sections_as_groupboxes(type(self.initial_settings), self.basic_layout)
        self.setLayout(main_layout)



    def _save_settings(self):
        """
            Save the current settings to the settings file.
        """
        # Plan is to save in settings and settings_handler pulls it from there
        print("Settings saved (Apply clicked).")

    def _save_and_close(self):
        """
            Save the current settings and close the dialog.
        """
        self._save_settings()
        self.accept()

    def _add_sections_as_groupboxes(self, settings_class: type[BaseModel], layout: QVBoxLayout):
        """
            Add sections of settings as group boxes.
        """
        for section_name, field in settings_class.model_fields.items():
            print(f"Adding section: {section_name}, Fields: {field}")

            if section_name in ["version"]:
                continue
            field_type = field.annotation

            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                group_box = QGroupBox(section_name)
                group_box.setLayout(QVBoxLayout())
                layout.addWidget(group_box)
                self._add_sections_as_groupboxes(field_type, group_box.layout())
            else:
                print(f"Dev | Adding field: {section_name} of type {field_type} to layout {layout}")
                # Create a field for the setting
                field_info = settings_class.model_fields[section_name]
                widget = create_input_widget_for_field(field_info)
                widget.setObjectName(section_name)
                # Create label with title or fallback to section_name and add widget next to it
                title = field_info.title or section_name
                label = QLabel(title)
                row = QHBoxLayout()
                row.addWidget(label)
                row.addWidget(widget)
                layout.addLayout(row)



    def _add_subsettings_as_fields(self, settings, layout: QVBoxLayout):
        """
            Recursively add subsettings as fields in the layout.
        """
        print(f"Dry | Adding subsettings for: {settings}")



def create_input_widget_for_field(field_info):
    default = field_info.get_default()  # default value if any
    desc = field_info.description or ""  # description for tooltip
    field_type = field_info.annotation  # type of the field


    if field_type == bool:
        widget = QCheckBox()
        if default:
            widget.setChecked(default)
    elif field_type == int:
        widget = QSpinBox()
        min_val, max_val = extract_bounds(field_info)
        widget.setRange(min_val, max_val)  # Example range, Change to metadata
        if default is not None:
            widget.setValue(default)
    elif field_type == float:
        widget = QDoubleSpinBox()
        if default is not None:
            widget.setValue(default)
    elif isinstance(field_info.annotation, type) and issubclass(field_info.annotation, Enum):
        # Enum field
        widget = QComboBox()
        choices = [e.name for e in field_info.annotation]  # or e.value
        widget.addItems(choices)
        if default:
            widget.setCurrentText(default.name if hasattr(default, "name") else str(default))
    else:
        # Fallback to QLineEdit for str or other types
        widget = QLineEdit()
        if default is not None:
            widget.setText(str(default))
    if desc:
        widget.setToolTip(desc)
    return widget

def extract_bounds(field_info: FieldInfo):
    min_val = None
    max_val = None

    for meta in field_info.metadata:
        if isinstance(meta, Ge):
            min_val = meta.ge
        elif isinstance(meta, Gt):
            min_val = meta.gt + 1
        elif isinstance(meta, Le):
            max_val = meta.le
        elif isinstance(meta, Lt):
            max_val = meta.lt - 1
    return min_val, max_val