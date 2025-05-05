import pathlib
from tkinter.messagebox import showinfo
from typing import List

import numpy as np
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QPushButton, QProgressBar
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLayout, QLabel, QGroupBox

from napari_pitcount_cfim.cellpose_analysis.cellpose_user import CellposeUser
from napari_pitcount_cfim.config.settings_handler import SettingsHandler
from napari_pitcount_cfim.image_handling.image_handler import ImageHandler
from napari_pitcount_cfim.loggers import setup_python_logging, setup_thread_exception_hook, qt_message_logger
from napari_pitcount_cfim.result_handling.result_handler import ResultHandler
from napari_pitcount_cfim.segmentation_worker import SegmentationWorker


class MainWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        # setup_python_logging()
        # qInstallMessageHandler(qt_message_logger)
        # setup_thread_exception_hook()


        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self)
        self.image_handler = ImageHandler(parent=self, napari_viewer=self.viewer)
        self.result_handler = ResultHandler(parent=self)
        self._workers = []

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

        pane = QGroupBox(self)
        pane.setTitle("Analysis")
        pane.setLayout(QVBoxLayout())
        self.analysis_button = QPushButton("Cellpose all images")
        self.analysis_button.clicked.connect(self._run_analysis)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)


        pane.layout().addWidget(self.analysis_button)
        pane.layout().addWidget(self.progress_bar)

        self.layout().addWidget(pane)


        self._update_widget_settings()


    def _update_widget_settings(self):
        """
        Update the settings of the widget.
        """
        settings = self.setting_handler.get_updated_settings()

        self.image_handler.set_output_path(settings.get("input_folder"))

        self.result_handler.set_output_path(settings.get("output_folder"))


    def _add_logo(self):
        """
        Add the logo to the widget.
        """
        path = pathlib.Path(__file__).parent / "logo" / "CFIM_logo_small.png"
        logo_label = QLabel()
        logo_label.setText(f"<img src='{path}' width='320'/>")
        self.layout().addWidget(logo_label)

    def _run_analysis(self):
        """Run Cellpose segmentation on all images using background threads."""
        layers = self.image_handler.get_all_images()
        total = len(layers)

        if total == 0:
            showinfo("No images loaded")
            return  # No images loaded, nothing to do

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        # Turn off the analysis button
        self.analysis_button.setEnabled(False)



        # Initialize counter for completed images
        self._completed = 0
        scale = self.image_handler.get_scale(0)
        cellpose_settings = self.setting_handler.get_updated_settings().get("cellpose_settings")
        if scale.shape == (3,):
            scale = scale[1:]

        # Define a slot to handle results coming from worker threads
        def _on_segmentation_result(mask, image_name):
            """Receive segmentation result from a worker and update the viewer/UI."""
            # Add the segmentation mask as a labels layer (only mask is added, no flows)
            self.viewer.add_labels(mask, name=f"{image_name}_mask", scale=scale)
            # Update progress
            self._completed += 1
            self.progress_bar.setValue(self._completed)
            # If all images are processed, ensure progress bar reaches 100%
            if self._completed == total:
                self.progress_bar.setValue(total)
                self.analysis_button.setEnabled(True)

        # Launch a worker thread for each image to run Cellpose in parallel
        for data in layers:
            # If layers are Napari layer objects, get the numpy data and name
            image_name = getattr(data, "name", "Image")  # layer.name if available
            cellpose_user = CellposeUser(cellpose_settings=cellpose_settings)

            worker = SegmentationWorker(data, image_name, cellpose_user)
            worker.result.connect(_on_segmentation_result)
            worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

            self._workers.append(worker)
            worker.start()



    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()