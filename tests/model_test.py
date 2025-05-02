import importlib.resources as pkg_resources

from napari_pitcount_cfim.ml_users.czseg_user import CZSegModel

pkg_root = pkg_resources.files("src")

# test_czseg_model.py

import tifffile
import napari

# Paths
model_path = pkg_root.joinpath("resources","models",'modelid=ac152c5c-b616-431c-ae2a-ea0ec307ce76')
tif_path = pkg_root.joinpath("resources","test_files", "P06 1-Tile-16-1-channel.tiff")

# Load image
image = tifffile.imread(tif_path)

# Initialize model
model = CZSegModel(model_path)

# Run prediction
label_map = model.predict(image)

# Visualize with napari
viewer = napari.Viewer()
viewer.add_image(image, name='Input Image')
viewer.add_labels(label_map, name='Prediction')
napari.run()
