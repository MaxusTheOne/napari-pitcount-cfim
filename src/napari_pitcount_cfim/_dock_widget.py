import os
from pathlib import Path
from tkinter.messagebox import showinfo
from typing import List

import cellpose
import napari
import numpy as np
from qtpy.QtCore import qInstallMessageHandler, QEventLoop
from qtpy.QtWidgets import QPushButton, QProgressBar
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLayout, QLabel, QGroupBox

from napari_pitcount_cfim.cellpose_analysis.cellpose_user import CellposeUser
from napari_pitcount_cfim.config.settings_handler import SettingsHandler
from napari_pitcount_cfim.image_handling.image_handler import ImageHandler
from napari_pitcount_cfim.loggers import setup_python_logging, setup_thread_exception_hook, qt_message_logger
from napari_pitcount_cfim.pitcounter.predict_user import ModelUser
from napari_pitcount_cfim.result_handling.result_handler import ResultHandler
from napari_pitcount_cfim.segmentation_worker import SegmentationWorker

# Default values
DEFAULT_MODEL = "ne64_md20_fl0.3"


class MainWidget(QWidget):
    def __init__(self, napari_viewer: napari.viewer, parent=None):
        super().__init__(parent=parent)

        # setup_python_logging()
        # qInstallMessageHandler(qt_message_logger)
        # setup_thread_exception_hook()
        if os.getenv("PITCOUNT_CFIM_NO_GUI", "0") == "1":
            self.no_gui = True
            self.verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))
        else:
            self.no_gui = False

        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self) #1
        self.image_handler: ImageHandler = ImageHandler(parent=self, napari_viewer=self.viewer, settings_handler=self.setting_handler)
        self.result_handler = ResultHandler(parent=self)
        self._workers = []
        self.model_user: ModelUser | None = None

        if self.no_gui:
            if self.verbosity > 0:
                print("NO GUI | Skipping GUI initialization.")
            self._run_pipeline()
        else:
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

            self.cellpose_button = QPushButton("Cellpose all images")
            self.cellpose_button.clicked.connect(self._run_cellpose_segmentation)

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setMinimum(0)

            self.ml_button = QPushButton("ML all images")
            self.ml_button.clicked.connect(self._run_ml_analysis)

            pane.layout().addWidget(self.cellpose_button)
            pane.layout().addWidget(self.progress_bar)
            pane.layout().addWidget(self.ml_button)

            self.layout().addWidget(pane)

            # self._update_widget_settings()

    def _run_pipeline(self):
        """
        Run the pipeline without GUI.
        """
        settings = self.setting_handler.get_updated_settings()

        # Load images
        input_path = os.getenv("PITCOUNT_CFIM_INPUT_FOLDER", "")
        verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))
        self.image_handler.load_images({"input_folder": input_path, "verbosity": verbosity})

        if self.verbosity >= 2:
            print(f"Loaded {len(self.image_handler.get_all_images())} images")
        # Run Cellpose analysis
        self._run_cellpose_segmentation()
        segmentation_masks = len(self.image_handler.get_all_labels())
        if self.verbosity >= 2:
            print(f"Completed Cellpose segmentation on {segmentation_masks} images")

        # Pit finder ml
        model_folder = os.getenv("PITCOUNT_CFIM_MODEL_FOLDER", None)
        self._run_ml_analysis(model_folder)
        if self.verbosity >= 2:
            print(f"Completed ML for {len(self.image_handler.get_all_labels()) - segmentation_masks} images")
        #
        # # Save results
        # self.result_handler.save_results()
        #
        print("Pipeline completed successfully.")
        self.viewer.close()

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
        path = Path(__file__).parent / "logo" / "CFIM_logo_small.png"
        logo_label = QLabel()
        logo_label.setText(f"<img src='{path}' width='320'/>")
        self.layout().addWidget(logo_label)


    def _run_estimate(self, image: np.ndarray = None):
        """
            Mostly for testing, runs Cellpose SizeModel to estimate diameter.
        """
        cp_version = cellpose.version
        if cp_version >= "4.0.1":
            print(f"Cellpose version {cp_version} does not support size estimation pre-analysis.\n setting diameter to 30.")
            return 30.0
        user = CellposeUser(cellpose_settings=self.setting_handler.get_settings().get("cellpose_settings"))
        diam = user.estimate_size(image)
        if self.verbosity >= 1:
            print(f"Estimated diameter: {diam}")

        return diam



    def _run_cellpose_segmentation(self):
        """Run Cellpose segmentation on all images using background threads."""
        layers = self.image_handler.get_all_images()
        total = len(layers)

        gui = not self.no_gui
        verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))

        if total == 0:
            showinfo("No images loaded")
            return  # No images loaded, nothing to do

        if gui:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")

            # Turn off the analysis button
            self.cellpose_button.setEnabled(False)
            self.cellpose_button.setText(f"Analyzing {total} images...")
        else:
            print(f"Running Cellpose on {total} images... | Dev Verbosity: {verbosity}")

            self._event_loop = QEventLoop()


        # Initialize counter for completed images
        self._completed = 0


        scale = self.image_handler.get_scale(0)
        cellpose_settings = self.setting_handler.get_updated_settings().get("cellpose_settings")

        if cellpose_settings.get("diameter") in (None, "", "0", 0.0, 0):
            cellpose_settings["diameter"] = self._run_estimate(image=layers[0])
        if scale.shape == (3,):
            scale = scale[1:]

        # Define a slot to handle results coming from worker threads
        def _on_segmentation_result(mask, image_name):
            """Receive segmentation result from a worker and update the viewer/UI."""
            self.image_handler.add_label(mask, name=f"{image_name}_mask", scale=scale, metadata={"cfim_type": "segmentation"})

            self._completed += 1
            if gui:
                # Update progress
                self.progress_bar.setValue(self._completed)
                if self._completed == total:
                    self.progress_bar.setValue(total)
                    self.cellpose_button.setEnabled(True)
                    self.cellpose_button.setText("Cellpose all images")
            else:
                if verbosity >= 1:
                    print(f"Completed {self._completed}/{total} images.")

                if self._completed == total:
                    if verbosity >= 1:
                        print("Cellpose analysis completed for all images.")
                    self._event_loop.quit()

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
        if not gui:
            self._event_loop.exec_()

    def _run_ml_analysis(self, model_folder: str = None):

        settings = self.setting_handler.get_updated_settings().get("model_settings")
        gui = not self.no_gui
        if model_folder:
            model_folder = Path(model_folder)
            if model_folder.exists():
                settings["model_folder"] = str(model_folder)
            else:
                print(f"No model folder found at {model_folder}")
                model_folder = None


        if settings["model_folder"] == "none" or not Path(settings["model_folder"]).exists() and not model_folder:
            print(f"Expected path to folder, got {settings["model_folder"]}, attempting to load default model.")
            settings["model_folder"] = Path(__file__).parent / "pitcounter" / "models" / DEFAULT_MODEL
            model_folder = settings["model_folder"]

            if not model_folder.exists():
                print(f"Getting default model at {model_folder} failed, exiting pit counting.")
                return
            self.setting_handler.update_settings("model_settings.model_folder", str(model_folder))

        self.model_user = ModelUser(model_folder=settings["model_folder"], prediction_settings=settings)


        images = self.image_handler.get_all_images_props(["data", "name", "uuid"])

        if gui:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(len(images))
            self.progress_bar.setValue(0)

            self.ml_button.setEnabled(False)

        completed = 0

        for image in images:
            data = image["data"]
            name = image["name"]
            unique_id = image["unique_id"]
            prediction = self.model_user.predict_from_npy(data)

            completed += 1
            self.image_handler.add_label(prediction, name=f"{name}_prediction", metadata={"cfim_type": "prediction", "image_input_id": unique_id})
            if gui:
                self.progress_bar.setValue(completed)
            else:
                if self.verbosity >= 1:
                    print(f"Completed {completed}/{len(images)} images.")
        if gui:
            self.ml_button.setEnabled(True)

    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()