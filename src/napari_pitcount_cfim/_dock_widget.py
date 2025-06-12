import os
import sys
import traceback
from pprint import pprint
from uuid import UUID
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

        ## Enable to log just about everything to the console.
        if os.getenv("PITCOUNT_CFIM_FULL_LOGGING", "0") == "1":
            setup_python_logging()
            qInstallMessageHandler(qt_message_logger)
            setup_thread_exception_hook()


        self.viewer = napari_viewer
        self.setting_handler = SettingsHandler(parent=self) #1
        self.image_handler: ImageHandler = ImageHandler(parent=self, napari_viewer=self.viewer, settings_handler=self.setting_handler)
        self.result_handler = ResultHandler(parent=self, settings_handler=self.setting_handler)
        self._workers = []
        self.model_user: ModelUser | None = None


        if os.getenv("PITCOUNT_CFIM_NO_GUI", "0") == "1":
            self.no_gui = True
            self.verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))
            try:
                if self.verbosity > 0:
                    print("NO GUI | Skipping GUI initialization.")
                self._run_pipeline()
            except Exception:
                traceback.print_exc()
                sys.exit(1)
        else:
            self.no_gui = False
            self.verbosity = 0

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

            self.ml_button = QPushButton("Run pit segmentation for all images")
            self.ml_button.clicked.connect(self._run_ml_analysis)

            pane.layout().addWidget(self.cellpose_button)
            pane.layout().addWidget(self.progress_bar)
            pane.layout().addWidget(self.ml_button)

            self.layout().addWidget(pane)

            pane = QGroupBox(self)
            pane.setTitle("Results")
            pane.setLayout(QVBoxLayout())

            self.results_button = self.result_handler.init_output_button_ui()

            pane.layout().addWidget(self.results_button)

            self.layout().addWidget(pane)

            # self._update_widget_settings()

    def _run_pipeline(self):
        """
        Run the pipeline without GUI.
        """
        settings = self.setting_handler.get_updated_settings()

        # Load images
        input_paths = os.getenv("PITCOUNT_CFIM_INPUT_FOLDER", "find_path")
        output_folder = os.getenv("PITCOUNT_CFIM_OUTPUT_FOLDER", "find_path")
        verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))
        save_raw = os.getenv("PITCOUNT_CFIM_SAVE_RAW_DATA", "0") == "1"
        dry_run = os.getenv("PITCOUNT_CFIM_DRY_RUN", "0") == "1"
        family_grouping = os.getenv("PITCOUNT_CFIM_FAMILY_GROUPING", "default")

        if not input_paths in ("find_path", "None", None, ""):
            input_paths = input_paths.split(";")


        image_layers = self.image_handler.load_images(input_folders = input_paths, verbosity = verbosity)
        for layer in image_layers:
            image_uuid = layer.unique_id
            self.result_handler.start_result_record(image_uuid=image_uuid, image_name=layer.name, folder_group=layer.metadata.get("folder_group", "default"))
            self.result_handler.set_image(image_uuid ,layer.data)

        if self.verbosity >= 2:
            print(f"NO GUI | Loaded {len(image_layers)} images")

        ## Run Cellpose analysis
        self._run_cellpose_segmentation(image_layers=image_layers)
        #region: Pit masks
        pit_mask_folder = os.getenv("PITCOUNT_CFIM_PIT_MASK_FOLDER", "None")

        if pit_mask_folder != "None":
            import czifile

            pit_mask_folder = Path(pit_mask_folder)
            if not pit_mask_folder.exists():
                raise ValueError(f"Provided pit mask folder '{pit_mask_folder}' does not exist.")
            if self.verbosity >= 2:
                print(f"NO GUI | Using provided pit mask folder: {pit_mask_folder}")
            # Load pit masks from the provided folder
            # TODO: Open using naparis .open() method
            pit_masks = []
            for ext in ["*.npy", "*.tif", "*.czi"]:
                for pit_mask_file in pit_mask_folder.glob(ext):
                    if ext == "*.czi":
                        pit_mask = np.squeeze(czifile.imread(pit_mask_file))
                    else:
                        pit_mask = np.load(pit_mask_file)
                    pit_masks.append(pit_mask)
                    file_name = pit_mask_file.stem
                    name = self.result_handler.set_pit_mask_by_name(file_name, pit_mask)
                    if self.verbosity >= 2:
                        print(f"NO GUI | Loaded pit mask '{file_name}' and associated it with image '{name}'.")

            if self.verbosity >= 2:
                print(f"NO GUI | Loaded {len(pit_masks)} pit masks from {pit_mask_folder}")
        else:
            # Pit finder ml
            model_folder = os.getenv("PITCOUNT_CFIM_MODEL_FOLDER", None)
            self._run_ml_analysis(model_folder)
            if self.verbosity >= 2:
                print(f"NO GUI | Completed ML")
        #endregion

        self.result_handler.set_raw_setting(save_raw)

        self.result_handler.create_and_save_results(output_folder, family_grouping=family_grouping)

        print("NO GUI | Pipeline completed successfully.")
        sys.exit(0)

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
        path = Path(__file__).parent.parent / "resources" / "logo" / "CFIM_logo_small.png"
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



    def _run_cellpose_segmentation(self, image_layers = None):
        """Run Cellpose segmentation on all images using background threads."""
        if image_layers in (None, False):
            layers = self.image_handler.get_all_images_props(["data", "name", "uuid"])
            self.result_handler.start_result_record_gui(layers)
            self.result_handler.set_images(layers)
        else:
            layers = image_layers
        total = len(layers)

        gui = not self.no_gui
        verbosity = int(os.getenv("PITCOUNT_CFIM_VERBOSITY", "0"))

        if total == 0:
            showinfo("No images loaded")
            return

        if gui:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")

            self.cellpose_button.setEnabled(False)
            self.cellpose_button.setText(f"Analyzing {total} images...")
        else:
            print(f"Running Cellpose on {total} images... ")

            self._event_loop = QEventLoop()

        self._completed = 0

        scale = self.image_handler.get_scale(0)
        cellpose_settings = self.setting_handler.get_updated_settings().get("cellpose_settings")

        if cellpose_settings.get("diameter") in (None, "", "0", 0.0, 0):
            cellpose_settings["diameter"] = self._run_estimate(image=layers[0].data)
        if scale.shape == (3,):
            scale = scale[1:]

        ## Callback function for thread workers.
        def _on_segmentation_result(mask, image_layer_name, image_uuid: UUID):
            """Receive segmentation result from a worker and update the viewer/UI."""
            self.result_handler.set_cell_mask(image_uuid, mask)
            self.image_handler.add_label(mask, name=f"{image_layer_name}_mask", scale=scale, metadata={"cfim_type": "segmentation"})

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

        ## Thread worker assignment.
        for layer in layers:
            data = layer.data
            name = layer.name
            uuid = layer.unique_id

            # napari_pitcount_cfim/_dock_widget:MainWidget._run_cellpose_segmentation #292
            cellpose_user = CellposeUser(cellpose_settings=cellpose_settings)

            worker = SegmentationWorker(data, name, uuid, cellpose_user)
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


        image_layers = self.image_handler.get_all_images_props(["data", "name", "uuid"])
        self.result_handler.start_result_record_gui(image_layers)

        if gui:
            total = len(image_layers)

            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.ml_button.setEnabled(False)
            self.cellpose_button.setText(f"Analyzing {total} images...")


        predictions = []
        completed = 0

        for layer in image_layers:
            data = layer.get("data")
            name = layer.get("name")
            unique_id = layer.get("unique_id")
            prediction = self.model_user.predict_from_npy(data)

            completed += 1
            self.image_handler.add_label(prediction, name=f"{name}_prediction", metadata={"cfim_type": "prediction", "image_input_id": unique_id})
            if gui:
                self.progress_bar.setValue(completed)
            else:
                predictions.append(prediction)
                if self.verbosity >= 1:
                    print(f"Completed {completed}/{len(image_layers)} images.")
            self.result_handler.set_pit_mask(image_uuid=unique_id, pit_mask=prediction)
        if gui:
            self.ml_button.setEnabled(True)
            self.ml_button.setText("Run pit segmentation for all images")



    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()