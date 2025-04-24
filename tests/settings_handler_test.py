import pytest
from pathlib import Path
from napari_pitcount_cfim.config.settings_handler import SettingsHandler

@pytest.fixture(autouse=True)
def no_side_effects(tmp_path, monkeypatch):
    # isolate filesystem, unset LOCALAPPDATA
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    yield

# Tests initialization with custom path
@pytest.mark.usefixtures("qapp")
def test_init_with_custom_path(monkeypatch, capsys, tmp_path):
    test_path = tmp_path / "mysettings"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    handler = SettingsHandler(path=str(test_path), debug=True)
    # folder path should be expanded to custom
    assert Path(handler.settings_folder_path) == test_path.expanduser()
    captured = capsys.readouterr()
    assert "Debug | Settings folder path" in captured.out

