import os
import sys
import subprocess
import yaml
from QtPy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QPushButton


class SettingsWidget(QWidget):
    def __init__(self, path=None, parent=None):
        super().__init__(parent)
        # Determine folder for settings
        self.settings_folder = os.path.expanduser(path) if path else os.getcwd()

        # Settings file name and full path
        self.settings_name = "settings.yaml"
        self.settings_file_path = os.path.join(self.settings_folder, self.settings_name)

        # Load or create
        self.settings = {}
        self._load_settings()

    def init_ui(self):
        pane = QGroupBox(self)
        pane.setTitle("Settings")
        pane.setLayout(QVBoxLayout())

        open_btn = QPushButton("Open Settings File")
        open_btn.clicked.connect(self.open_settings)
        pane.layout().addWidget(open_btn)

        return pane

    def open_settings(self):
        """Launch the YAML file in the default editor."""
        fp = self.settings_file_path
        if sys.platform == "win32":
            os.startfile(fp)
        elif sys.platform == "darwin":
            subprocess.call(["open", fp])
        else:
            subprocess.call(["xdg-open", fp])

    def update_settings(self):
        """Reload from disk and return the raw dict."""
        self._load_settings()
        return self.settings

    def get_settings(self):
        """Return the current settings dict (in memory)."""
        return self.settings

    def _load_settings(self):
        """Load settings from YAML; if missing or invalid, create defaults."""
        if os.path.exists(self.settings_file_path):
            try:
                with open(self.settings_file_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                self.settings = data
                return
            except yaml.YAMLError as e:
                print(f"[!] Could not parse {self.settings_file_path}: {e}")

        # Fallback: make a new settings file
        self._make_settings_file()

    def _make_settings_file(self):
        """Create a default YAML file (empty dict) if none exists."""
        os.makedirs(self.settings_folder, exist_ok=True)
        self.settings = {}
        self._save_settings()

    def _save_settings(self):
        """Write current settings to disk."""
        with open(self.settings_file_path, "w") as f:
            yaml.dump(self.settings, f, sort_keys=False)
