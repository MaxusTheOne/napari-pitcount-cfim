# Python
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QFileDialog

class ResultHandler(QWidget):
    """
    A class to handle and output result dictionaries as plain text files.
    """
    def __init__(self, parent=None, output_path=None, prompt_for_folder=True):
        super().__init__(parent)
        self.output_path = output_path
        self.prompt_for_folder = prompt_for_folder

    def set_output_path(self, path):
        """
        Set the output path where result files will be saved.
        """
        if not isinstance(path, str):
            raise ValueError("Output path must be a string.")
        self.output_path = path

    def _select_folder(self) -> bool:
        """
        Opens a folder selection dialog and updates the output path.
        Returns False if the user cancels.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.output_path) if self.output_path else ""
        )
        if not folder:
            return False
        self.output_path = folder
        return True

    def output_results(self, results: dict):
        """
        Take results in the form of a dictionary where each key is a name and the value is a result dictionary.
        Outputs each result as plain text to a separate file named after the key.
        """
        if self.prompt_for_folder:
            if not self._select_folder():
                return  # user cancelled folder selection

        if not self.output_path:
            raise ValueError("Output path is not set. Please set the output path before outputting results.")

        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, result in results.items():
            file_path = output_dir / f"{name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(result))