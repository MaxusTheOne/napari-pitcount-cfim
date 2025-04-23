from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit

class MainWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        print(f"[*] Initializing MainWidget")
        self.viewer = napari_viewer

        layout = QVBoxLayout()
        self.setLayout(layout)

        hello_text_field = QLineEdit("Hello, World!")

        self.layout().addWidget(hello_text_field)

