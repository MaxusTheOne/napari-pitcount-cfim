# Python
import json
import uuid
from pathlib import Path

import numpy as np
# removed unused plt import
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QWidget, QFileDialog, QPushButton
from skimage import measure
from skimage.measure import regionprops

from napari_pitcount_cfim.image_handling.image_handler import ImageLayerProps
from napari_pitcount_cfim.result_handling.graphs import simple_graph, modern_graph

# DEFAULTS
MIN_RESULTS_OR_SPLIT = 3

class ResultHandler(QWidget):
    """
        A class to handle and output results
    """
    def __init__(self, parent=None, settings_handler=None):
        super().__init__(parent)
        self.settings_handler = settings_handler
        self.settings_parent = settings_handler.get_settings()
        self.settings = self.settings_parent["file_settings"]
        self.output = self.settings["output_folder"]
        self.prompt_for_folder = self.settings["folder_prompt"]
        self.skip_output_confirmation = False
        self.debug = self.settings.get("debug", True)

        self.results = {}
        self._results = {}
        self.result_by_cell = {}

        self.output_button = None

    def set_output_path(self, path):
        """
            Set the output path where result files will be saved.
        """
        if not isinstance(path, str):
            raise ValueError("Output path must be a string.")
        self.output = path

    def init_output_button_ui(self):
        self.output_button = QPushButton("Get results")
        self.output_button.clicked.connect(self.create_and_save_results)


        return self.output_button

    def _select_folder(self) -> bool:
        """
            Opens a folder selection dialog and updates the output path.
            Returns False if the user cancels.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.output) if self.output else ""
        )
        if not folder:
            return False
        self.output = folder
        return True
    def set_raw_setting(self, value: bool):
        """
            Set the raw data saving setting.
            If True, raw data will be saved to the output folder.
        """
        if not isinstance(value, bool):
            raise ValueError("Raw data setting must be a boolean.")
        self.settings["save_raw_data"] = value


    def create_and_save_results(self, output_path: str = None, family_grouping: str = "Default"):

        # legacy: if passed a dict of results, output simple text files
        if isinstance(output_path, dict):
            return self._output_results(results=output_path)

        settings = self.settings_handler.get_updated_settings().get("file_settings", {})
        if not output_path:
            output_path = settings.get("output_folder", "find_path")
        if family_grouping == "Default":
            family_grouping = settings.get("family_grouping", "Default").lower()
        else:
            family_grouping = family_grouping.lower()
        if not output_path == "find_path":
            self.set_output_path(output_path)
            print(f"Results | Output set to {self.output}")

        results = self.get_valid_results()
        # precompute per-image statistics for fallback graph
        image_stats = {}
        for data in results.values():
            name = data['image_name']
            pits = self._extract_pits_from_mask(data['pit_masks'])
            cells = self.filter_pit_to_cells(pit_data=pits, cell_mask=data['cell_masks'])
            image_stats[name] = self._generate_statistics(cells)

        stats = []
        # compute and save grouped statistics
        grouped_cells = {}
        names_by_group = {}
        for data in results.values():
            image_name = data["image_name"]
            group_key = None
            if family_grouping in ("folder",):
                group_key = data.get("folder_group") or "Default"
            elif family_grouping in ("analysis", "all"):
                group_key = "all"
            else:  # file or default
                group_key = image_name

            # extract and assign pits to cells
            pits = self._extract_pits_from_mask(data["pit_masks"])
            cells = self.filter_pit_to_cells(pit_data=pits, cell_mask=data["cell_masks"])
            grouped_cells.setdefault(group_key, []).append(cells)
            names_by_group.setdefault(group_key, []).append(image_name)

        # compute and save grouped statistics
        for group, cell_lists in grouped_cells.items():
            # aggregate stats
            total_cells = sum(len(c) for c in cell_lists)
            total_pits = sum(sum(len(d["pits"]) for d in c.values()) for c in cell_lists)
            cells_with_pits = sum(sum(1 for d in c.values() if len(d["pits"])>0) for c in cell_lists)
            avg_pits = total_pits / total_cells if total_cells else 0.0
            pct_with = (cells_with_pits/total_cells*100) if total_cells else 0.0
            group_stats = {
                "group": family_grouping + " / " + group,
                "total_cells": total_cells,
                "total_pits": total_pits,
                "pits_per_cell_avg": avg_pits,
                "cells_with_pits": cells_with_pits,
                "cells_with_pits_percent": pct_with
            }

            # save raw data if enabled
            if self.settings.get("save_raw_data"):
                raw_path = Path(self.output) / f"raw_data_{group}.json"
                raw_dict = {name: cells for name, cells in zip(names_by_group[group], cell_lists)}
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(raw_dict, f, indent=2)

            # save stats
            stats_file = Path(self.output) / f"stats_{group}.json"
            stats.append(group_stats)
            self._save_statistics(stats, output_path=stats_file)

        # generate graph if stats exist
        if stats:
            # choose graph style from settings (simple or modern)
            graph_style = settings.get('graph_style', 'modern').lower()
            # determine graph data: fallback to per-file if too few folders
            if family_grouping == 'folder' and len(stats) <= MIN_RESULTS_OR_SPLIT:
                graph_label = 'file'
                graph_groups = list(image_stats.keys())
                avg_vals = [image_stats[g]['pits_per_cell_avg'] for g in graph_groups]
                pct_vals = [image_stats[g]['cells_with_pits_percent'] for g in graph_groups]
            else:
                graph_label = family_grouping
                graph_groups = [s['group'] for s in stats]
                avg_vals = [s['pits_per_cell_avg'] for s in stats]
                pct_vals = [s['cells_with_pits_percent'] for s in stats]
            # call the selected graph function
            if graph_style == 'modern':
                modern_graph(graph_label, avg_vals, pct_vals, graph_groups, self.output)
            else:
                simple_graph(graph_label, avg_vals, pct_vals, graph_groups, self.output)

        return stats

    def _save_statistics(self, stats: list, output_path: str | Path = None):
        """
            Save statistics to the output folder.
            Expects stats to be a list of dictionaries with summary statistics.
        """

        if not isinstance(stats, list):
            raise ValueError("Stats must be a list of dictionaries.")
        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_folder = output_path.parent
        output_folder.mkdir(parents=False, exist_ok=True)

        stats_file = output_path
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")


    def save_raw_data_to_output(self, raw_data: dict, image_name: str = "raw_image"):
        """
            Save raw data to the output folder.
            Expects raw_data to be a dictionary with cells{pits}.
        """
        if not self.settings.get("save_raw_data", False):
            print("Raw data saving is disabled. Set 'save_raw_data' to True in settings or in arguments.")
            return

        if not isinstance(raw_data, dict):
            raise ValueError("Raw data must be a dictionary.")


    def count_cell_list(self, cell_mask_list: list) -> list:

        total = 0
        for cell_mask in cell_mask_list:
            if not isinstance(cell_mask, np.ndarray):
                print(f"Cell mask {cell_mask} is not a numpy array.")
                raise ValueError("Each cell mask must be a numpy array.")
            total += self._count_cells(cell_mask)
        return total



    def count_cells(self, cell_mask: np.ndarray) -> int:
        return self._count_cells(cell_mask)

    def count_pits_list(self, pit_mask_list: list) -> tuple[list, int]:
        total_pits_list = []
        total = 0
        for pit_mask in pit_mask_list:
            pits = self._extract_pits_from_mask(pit_mask)
            total += len(pits)
            total_pits_list.append(pits)
        return total_pits_list, total


    def add_result(self, result_type: str, data):
        if not self.results.get(result_type):
            self.results[result_type] = [data]
        else:
            self.results[result_type].append(data)

    #region Strict result handling
    def start_result_record(self, image_uuid, image_name, folder_group=None):
        """
            Start recording results for a new image.
            Expects image_uuid and image_name to be unique identifiers for the image.
        """

        if not isinstance(image_uuid, uuid.UUID):
            raise ValueError("Image UUID must be a UUID")
        self._results[image_uuid] = {"image_name": image_name,"folder_group":folder_group ,"image_data": None, "pit_masks": None, "cell_masks": None}

    def start_result_record_gui(self, layers: list[ImageLayerProps], strict=False):
        """
            Start recording results for a new image using GUI layers.
            Expects layers to be a list of ImageLayerProps with unique UUIDs.
        """
        if not isinstance(layers, list):
            raise ValueError("Layers must be a list of ImageLayerProps")
        for layer in layers:
            if not isinstance(layer, ImageLayerProps):
                raise ValueError("Each layer must be an instance of ImageLayerProps")
            if layer.unique_id in self._results:
                if strict:
                    raise ValueError(f"Image UUID {layer.unique_id} already exists in results.")
                else:
                    continue
            self.start_result_record(layer.unique_id, layer.name)

    def get_valid_results(self) -> dict:
        """
            Get results that have all required fields set.
            Returns a dictionary with image UUIDs as keys and their data as values.
        """
        valid_results = {}
        for image_uuid, data in self._results.items():
            if data["image_data"] is not None and data["pit_masks"] is not None and data["cell_masks"] is not None:
                valid_results[image_uuid] = data
            else:
                if self.debug:
                    print(f"Debug | result_handler | Skipping result for {image_uuid} due to missing fields.")
        return valid_results

    def set_image(self, image_uuid: uuid.UUID, image_data: np.ndarray, strict: bool = False):
        """
            Set the image data for a specific image UUID.
            Expects image_data to be a numpy array.
        """
        if not isinstance(image_uuid, uuid.UUID):
            raise ValueError("Image UUID must be a UUID")
        if not isinstance(image_data, np.ndarray):
            raise ValueError("Image data must be a numpy array")
        if image_uuid not in self._results:
            raise KeyError(f"Image UUID {image_uuid} not found in results.")
        if self._results[image_uuid]["image_data"] is not None:
            if strict:
                raise ValueError(f"set_image | Strict | Image UUID {image_uuid} already has image data set")
            else:
                show_warning("Image data already set for this UUID. Overwriting it.")
        self._results[image_uuid]["image_data"] = image_data

    def set_images(self, image_layers: list[ImageLayerProps], strict: bool = False):
        """
            Set the image data for multiple images using a list of ImageLayerProps.
            Expects image_layers to be a list of ImageLayerProps with unique UUIDs.
        """
        if not isinstance(image_layers, list):
            raise ValueError("Image layers must be a list of ImageLayerProps")
        for layer in image_layers:
            if not isinstance(layer, ImageLayerProps):
                raise ValueError("Each layer must be an instance of ImageLayerProps")
            self.set_image(layer.unique_id, layer.data, strict=strict)



    def set_pit_mask(self, image_uuid: uuid.UUID, pit_mask: np.ndarray, strict: bool = False):
        """
            Set the pit mask for a specific image UUID.
            Expects pit_mask to be a numpy array.
        """
        if not isinstance(image_uuid, uuid.UUID):
            raise ValueError("Image UUID must be a UUID")
        if not isinstance(pit_mask, np.ndarray):
            raise ValueError("Pit mask must be a numpy array")
        if image_uuid not in self._results:
            raise KeyError(f"Image UUID {image_uuid} not found in results.")
        if self._results[image_uuid]["pit_masks"] is not None:
            if strict:
                raise ValueError(f"set_pit_mask | Strict | Image UUID {image_uuid} already has pit mask set")
            else:
                show_warning("Pit mask already set for this UUID. Overwriting it.")
        self._results[image_uuid]["pit_masks"] = pit_mask

    def set_pit_mask_by_name(self, image_name: str, pit_mask: np.ndarray):
        """
            Set the pit mask for a specific image name.
            Expects pit_mask to be a numpy array.
        """
        if not isinstance(image_name, str):
            raise ValueError("Image name must be a string")
        if not isinstance(pit_mask, np.ndarray):
            raise ValueError("Pit mask must be a numpy array")
        ## If image_name ends in _label, remove it
        if image_name.endswith("_label"):
            image_name = image_name[:-6]


        # Find the UUID by image name
        image_uuid = next((uuid for uuid, data in self._results.items() if image_name in data["image_name"]), None)
        if image_uuid is None:
            raise KeyError(f"Image name {image_name} not found in results: {list(self._results.keys())}")

        self._results[image_uuid]["pit_masks"] = pit_mask

        return self._results[image_uuid].get("image_name", "Unknown Image Name")

    def set_cell_mask(self, image_uuid: uuid.UUID, cell_mask: np.ndarray, strict: bool = False):
        """
            Set the cell mask for a specific image UUID.
            Expects cell_mask to be a numpy array.
        """
        if not isinstance(image_uuid, uuid.UUID):
            raise ValueError("Image UUID must be a UUID")
        if not isinstance(cell_mask, np.ndarray):
            raise ValueError("Cell mask must be a numpy array")
        if image_uuid not in self._results:
            raise KeyError(f"Image UUID {image_uuid} not found in results.")
        if self._results[image_uuid]["cell_masks"] is not None:
            if strict:
                raise ValueError(f"set_cell_mask | Strict | Image UUID {image_uuid} already has cell mask set")
            else:
                show_warning("Cell mask already set for this UUID. Overwriting it.")
        self._results[image_uuid]["cell_masks"] = cell_mask

    def dump_results(self):
        return self._results

    #endregion

    def set_result_layers(self, result_array_dict: dict[str, np.ndarray]):
        if not all(k in result_array_dict for k in ("images", "pit_masks", "cell_masks")):
            raise ValueError(f"Result layers must contain images, pit_masks and cell_masks. Got {result_array_dict.keys()} instead.")
        self.results["images"] = result_array_dict["images"]
        self.results["pit_masks"] = result_array_dict["pit_masks"]
        self.results["cell_masks"] = result_array_dict["cell_masks"]

    def filter_pit_to_cells(self, cell_mask: np.ndarray, pit_data):
        """
        Assign pits to the cell in which their centroid falls.

        Parameters:
        - pit_data (List[Dict]): List of pit dicts with keys "coords" and "size"
        - cell_mask (np.ndarray): 2D or 3D array with labeled cells

        Returns:
        - Dict[int, Dict]: A dictionary keyed by cell ID with:
            {
                "est_center": (z, y, x) or (y, x),
                "pits": [pit_dicts]
            }
        """
        cell_dict = {}

        # Measure properties of labeled cells
        props = regionprops(cell_mask)
        for prop in props:
            cell_id = prop.label
            centroid = tuple(map(int, prop.centroid))
            cell_dict[cell_id] = {
                "est_center": centroid,
                "pits": []
            }

        # Assign each pit to the corresponding cell
        for pit in pit_data:
            coords = pit["coords"]
            try:
                cell_id = int(cell_mask[coords])
            except IndexError:
                continue

            if cell_id > 0 and cell_id in cell_dict:
                cell_dict[cell_id]["pits"].append(pit)

        self.result_by_cell = cell_dict

        return cell_dict


    def _extract_pits_from_mask(self, pit_mask: np.ndarray):

        pit_data = []

        if pit_mask.ndim == 2:
            # 2D case
            labeled = measure.label(pit_mask, connectivity=1)
            props = measure.regionprops(labeled)
            for prop in props:
                centroid = tuple(map(int, prop.centroid))  # (y, x)
                pit_data.append({
                    "coords": centroid,
                    "size": prop.area
                })
        elif pit_mask.ndim == 3:
            print(f"3D unsopported for now. Pit mask has {pit_mask.shape} shape.")

        else:
            raise ValueError(f"Unexpected pit mask dimensions. Expected 2D, got {pit_mask.ndim}D.")

        return pit_data

    def _select_label(self, cell_mask: np.ndarray, label_num: int):
        """
            Selects a specific label from a binary mask.
            Returns: a boolean mask
        """

        return cell_mask == label_num

    def _crop_image_to_cell_mask(self, image: np.ndarray, cell_mask: np.ndarray):
        """
            Crops an image to the bounding box of the cell mask.
            Assumes the mask is a 2D numpy array with 1s for cells and 0s for background.

            returns: the cropped image and the cropped mask, and the bounding box
        """
        coords = np.argwhere(cell_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1  # +1 to include the endpoint

        # Crop the image and label
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = cell_mask[y_min:y_max, x_min:x_max]

        return cropped_image, cropped_mask, (y_min, x_min, y_max, x_max)

    def _count_cells(self, cell_mask: np.ndarray) -> int:
        """
            Count the number of cells in a binary mask.
            Assumes the mask is a 2D numpy array with 1s for cells and 0s for background.
        """
        labels = np.unique(cell_mask)
        labels = labels[labels != 0]  # remove background (0)
        num_labels = len(labels)

        return num_labels

    def _output_results(self, results: dict = None):
        """
            Take results in the form of a dictionary where each key is a name and the value is a result dictionary.
            Outputs each result as plain text to a separate file named after the key.
        """
        if results is None:
            results = self.results

        if self.prompt_for_folder:
            if not self._select_folder():
                return  # user cancelled folder selection

        if not self.output:
            raise ValueError("Output path is not set. Please set the output path before outputting results.")

        output_dir = Path(self.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, result in results.items():
            file_path = output_dir / f"{name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(result))

    def get(self, key: str):
        """
            Get a specific result by key.
            Returns None if the key does not exist.
        """
        return self.results.get(key, None)

    def _generate_statistics(self, cell_results: dict) -> dict:
        """
        Generate statistics from the per-cell results of a single image.

        Parameters:
        - cell_results (dict): The result of `create_cell_pit_results()` for one image

        Returns:
        - dict with summary statistics:
            {
                "total_cells": int,
                "total_pits": int,
                "pits_per_cell_avg": float,
                "cells_with_pits": int,
                "cells_with_pits_percent": float
            }
        """
        total_cells = len(cell_results)
        total_pits = sum(len(data["pits"]) for data in cell_results.values())
        cells_with_pits = sum(1 for data in cell_results.values() if len(data["pits"]) > 0)

        return {
            "total_cells": total_cells,
            "total_pits": total_pits,
            "pits_per_cell_avg": total_pits / total_cells if total_cells > 0 else 0.0,
            "cells_with_pits": cells_with_pits,
            "cells_with_pits_percent": (cells_with_pits / total_cells * 100) if total_cells > 0 else 0.0
        }
