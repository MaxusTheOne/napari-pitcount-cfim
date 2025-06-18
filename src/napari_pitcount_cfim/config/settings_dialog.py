from enum import Enum

from PyQt6.QtWidgets import QGroupBox, QLayout
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
        self.setLayout(main_layout)

        # Populate the tabs with settings fields
        container = QWidget()
        main_layout = QVBoxLayout(container)

        # populate everything into main_layout
        build_ui_from_model(self.initial_settings, main_layout)

        self.basic_layout.addWidget(container)



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

def build_ui_from_model(model: BaseModel, layout: QLayout):
    for name, field in type(model).model_fields.items():
        if name == "version":
            continue

        value = getattr(model, name)
        field_type = field.annotation

        # if it's a nested BaseModel, make a QGroupBox
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            groupbox = QGroupBox(name)
            groupbox_layout = QVBoxLayout()
            groupbox.setLayout(groupbox_layout)
            # recurse into the nested instance
            build_ui_from_model(value, groupbox_layout)
            layout.addWidget(groupbox)

        else:
            # simple field → label + input
            label = QLabel(field.title or name)
            widget = create_input_widget_for_field(field, value)
            widget.setObjectName(name)

            row = QHBoxLayout()
            row.addWidget(label)
            row.addStretch(1) ## Should align the label to the left
            row.addWidget(widget)
            layout.addLayout(row)



def create_input_widget_for_field(field_info: FieldInfo, value=None):


    def extract_bounds():
        min_val, max_val = None, None
        for meta in field_info.metadata:
            if isinstance(meta, Ge): min_val = meta.ge
            elif isinstance(meta, Gt): min_val = meta.gt + 1
            elif isinstance(meta, Le): max_val = meta.le
            elif isinstance(meta, Lt): max_val = meta.lt - 1
        return min_val, max_val

    typ = field_info.annotation
    desc = field_info.description or ""

    if typ is bool:
        w = QCheckBox()
        w.setChecked(bool(value))

    elif typ is int:
        w = QSpinBox()
        min_val, max_val = extract_bounds()
        w.setMinimum(min_val if min_val is not None else -2**31)
        w.setMaximum(max_val if max_val is not None else 2**31 - 1)
        w.setValue(value if value is not None else 0)

    elif typ is float:
        w = QDoubleSpinBox()
        min_val, max_val = extract_bounds()
        w.setMinimum(min_val if min_val is not None else -1e6)
        w.setMaximum(max_val if max_val is not None else 1e6)
        w.setValue(value if value is not None else 0.0)

    elif isinstance(typ, type) and issubclass(typ, Enum):
        w = QComboBox()
        options = [e.name for e in typ]
        w.addItems(options)
        if value is not None:
            w.setCurrentText(value.name if hasattr(value, "name") else str(value))

    else:
        w = QLineEdit()
        if value is not None:
            w.setText(str(value))

    if desc:
        w.setToolTip(desc)
    return w
