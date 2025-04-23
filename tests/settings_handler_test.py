import pytest
from unittest.mock import patch, mock_open

from napari_pitcount_cfim.config.settings_handler import SettingsHandler


@pytest.fixture
def settings_handler():
    return SettingsHandler(debug=True)

def loads_settings_file_when_exists(settings_handler):
    mock_data = {"version": "1.0", "output_folder": "test_folder"}
    with patch("builtins.open", mock_open(read_data="version: 1.0\noutput_folder: test_folder")):
        with patch("yaml.safe_load", return_value=mock_data):
            with patch("os.path.exists", return_value=True):
                settings_handler._load_settings()
                assert settings_handler.settings.output_folder == "test_folder"

def falls_back_to_default_settings_when_file_missing(settings_handler):
    with patch("os.path.exists", return_value=False):
        with patch("src.napari_pitcount_cfim.config.settings_handler.CFIMSettings") as MockCFIMSettings:
            mock_settings = MockCFIMSettings.return_value
            mock_settings.model_copy.return_value = {"output_folder": "default_folder"}
            settings_handler._load_settings()
            assert settings_handler.settings.output_folder == "default_folder"

def creates_new_settings_file_with_default_values(settings_handler):
    with patch("builtins.open", mock_open()) as mocked_file:
        with patch("yaml.dump") as mock_yaml_dump:
            settings_handler._make_settings_file()
            mocked_file.assert_called_once_with(settings_handler.settings_file_path, "w")
            mock_yaml_dump.assert_called_once_with(settings_handler.settings.model_dump(), mocked_file(), sort_keys=False)

def opens_settings_file_on_windows(settings_handler):
    with patch("os.startfile") as mock_startfile:
        with patch("sys.platform", "win32"):
            settings_handler.open_settings_file()
            mock_startfile.assert_called_once_with(settings_handler.settings_file_path)

def migrates_settings_when_version_is_outdated(settings_handler):
    mock_data = {"version": "0.9", "output_folder": "old_folder"}
    with patch("src.napari_pitcount_cfim.config.settings_handler.CFIMSettings") as MockCFIMSettings:
        mock_settings = MockCFIMSettings.return_value
        mock_settings.model_dump.return_value = {"version": "1.0", "output_folder": "default_folder"}
        updated_data, was_migrated = settings_handler.migrate_settings_if_needed(mock_data)
        assert was_migrated is True
        assert updated_data["version"] == "1.0"
        assert updated_data["output_folder"] == "old_folder"