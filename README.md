# napari-pitcount-cfim

## License
BSD 3-Clause

## About
This napari plugin was developed in partnership with CFIM (Centre for Microscopy and Image Analysis, Copenhagen University).

The plugin enables image analysis for microscopy, focused on identifying pits and segmenting cells, then generating detailed statistics. It is tailored for using `.czi` files and integrates well with the [`napari-czi-reader`](https://github.com/MaxusTheOne/napari-czi-reader).

For training the VGG19 2_2 × Random Forest Classifier used in this plugin, visit the [pitcount-ml-training](https://github.com/MaxusTheOne/pitcount-ml-training) repository.

## Features
- Detects pits in images using a trained `torchvision` model.
- Performs cell segmentation via Cellpose (default model: `cyto3`).
- Calculates and outputs statistics such as:
  - Total cell count
  - Total pit count
  - Percentage of cells containing pits
  - Average number of pits per cell

## Usage

### Graphical Mode (GUI)
You can launch the plugin in napari with:
```bash
napari-pitcount-cfim --dev
```
or open napari and activate the plugin manually.
## Headless Mode (NO GUI)
```bash
napari-pitcount-cfim --no-gui 
```
Run --help to list all options:
```bash
napari-pitcount-cfim --no-gui -h
```
## Command-Line Arguments
| Argument            | Alias | Type      | Description                                                                            |
| ------------------- | ----- | --------- | -------------------------------------------------------------------------------------- |
| `--no-gui`          |       | flag      | Runs the pipeline without GUI. Required for headless automation.                       |
| `--dev`             |       | flag      | Launches napari in developer mode for plugin debugging.                                |
| `--verbosity`       | `-v`  | int (0–2) | Sets the level of console output. Default: `0`.                                        |
| `--input-folder`    | `-i`  | str       | Input directory for image data (required with `--no-gui`).                             |
| `--output-folder`   | `-o`  | str       | Directory to save results. Default: `'output'`.                                        |
| `--pit-mask-folder` | `-p`  | path      | If specified, skips pit prediction and uses this directory for pit masks.              |
| `--save-raw-data`   |       | flag      | Saves raw, unprocessed data to the output folder (only in `--no-gui` mode).            |
| `--family-grouping` |       | str       | Grouping method for output: `default`, `file`, `folder`, or `all`. Default: `default`. |

## Notes
- --input-folder must be used with --no-gui.

- --pit-mask-folder must be a valid existing directory.

- Set environment variables are used internally to control behavior.

## Requirements
Napari recommends installing napari seperately, as it is not included in this package. You can install it with:
```bash
pip install napari[all]
```
Or you can just
```bash
pip install napari-pitcount-cfim[napari]
```

## Known Issues
- The plugin might not support the formats of most model output.
- It's not possible to link masks directly to images in the GUI.
- The default pit model, is a stub and mostly for decoration.









