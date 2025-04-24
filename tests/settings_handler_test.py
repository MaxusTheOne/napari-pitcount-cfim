import pytest
from pathlib import Path

import yaml

from napari_pitcount_cfim.config.settings_handler import SettingsHandler
from napari_pitcount_cfim.config.settings_structure import CFIMSettings


@pytest.fixture(autouse=True)
def temp_path_cleanup(tmp_path, monkeypatch):
    # isolate filesystem, unset LOCALAPPDATA
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    yield

# Test 0: initialization with custom path
@pytest.mark.usefixtures("qapp")
def test_init_with_custom_path(monkeypatch, capsys, tmp_path):
    test_path = tmp_path / "mysettings"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    handler = SettingsHandler(path=str(test_path), debug=True)

    # folder path should be expanded to custom
    assert Path(handler.settings_folder_path) == test_path.expanduser()
    captured = capsys.readouterr()
    assert "Debug | Settings folder path" in captured.out

# Test 1: Verify that a new settings file is created if it does not exist
@pytest.mark.usefixtures("qapp")
def test_create_settings_file(monkeypatch, tmp_path, capsys):

    # set a custom settings path
    test_dir = tmp_path / "mysettings"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    if test_dir.exists():

        # cleanup if already present
        for item in test_dir.iterdir():
            item.unlink()
        test_dir.rmdir()

    # Initialize SettingsHandler using the custom path
    handler = SettingsHandler(path=str(test_dir), debug=True)
    settings_file = Path(handler.settings_file_path)

    # Check if the new settings file was created
    assert settings_file.exists()

    # Load file content and check version has been set to newest version
    with open(settings_file, "r") as file:
        content = yaml.safe_load(file)
    assert content.get("version") == CFIMSettings.__version__
    captured = capsys.readouterr()
    assert "Debug | Settings folder path" in captured.out

# Test 2: Verify that an outdated settings file gets migrated
@pytest.mark.usefixtures("qapp")
def test_migrate_outdated_settings(monkeypatch, tmp_path):
    test_dir = tmp_path / "mysettings"
    test_dir.mkdir(exist_ok=True)

    # Build an outdated settings file with version "0.0"
    outdated_settings = CFIMSettings().model_dump()
    outdated_settings["version"] = "0.0"
    settings_file = test_dir / "napari_pitcount_cfim_settings.yaml"
    with open(settings_file, "w") as file:
        yaml.dump(outdated_settings, file, sort_keys=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    # Initialize SettingsHandler using the custom path
    handler = SettingsHandler(path=str(test_dir), debug=True)

    # After instantiation, the settings should have been migrated
    updated_settings = handler.get_updated_settings()
    assert updated_settings.get("version") == CFIMSettings.__version__