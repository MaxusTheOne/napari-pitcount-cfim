import pytest
from PyQt5.QtWidgets import QApplication

from napari_pitcount_cfim.result_handling.result_handler import ResultHandler


@pytest.fixture
def qapp():
    """Fixture for a Qt application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

def test_output_results_creates_files(tmp_path, qapp):
    # Instantiate the ResultHandler with prompt disabled and set output_path
    handler = ResultHandler(output_path=str(tmp_path), prompt_for_folder=False)
    dummy_results = {
        "test": {"value": 123},
        "another": {"value": "abc"}
    }
    handler.output_results(dummy_results)

    # Verify the output files are created and contain expected content
    test_file = tmp_path / "test.txt"
    another_file = tmp_path / "another.txt"
    assert test_file.exists()
    assert another_file.exists()

    with open(test_file, "r", encoding="utf-8") as f:
        content_test = f.read()
    with open(another_file, "r", encoding="utf-8") as f:
        content_another = f.read()

    assert "123" in content_test
    assert "abc" in content_another

def test_set_output_path_invalid(qapp):
    # Verify that setting a non-string output path raises ValueError
    handler = ResultHandler(prompt_for_folder=False)
    with pytest.raises(ValueError):
        handler.set_output_path(123)

def test_output_results_no_output_path(qapp):
    # Verify that outputting results without an output path set raises ValueError
    handler = ResultHandler(prompt_for_folder=False)
    dummy_results = {"test": {"value": 123}}
    with pytest.raises(ValueError):
        handler.output_results(dummy_results)