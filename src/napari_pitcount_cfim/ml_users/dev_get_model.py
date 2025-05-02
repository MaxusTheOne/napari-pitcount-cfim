import zipfile, h5py
from pathlib import Path

path = Path(r"C:\napari-pitcount-cfim\src\resources\models\flour_model\ac152c5c-b616-431c-ae2a-ea0ec307ce76.model")
czseg_file = Path(r'C:\napari-pitcount-cfim\src\resources\models\Frederikke.pit.fluor.czseg')

print("Is a ZIP archive?", zipfile.is_zipfile(czseg_file))

try:
    with h5py.File(czseg_file, "r") as f:
        print("Looks like an HDF5 file (Keras).")
except OSError:
    print("Not an HDF5 file.")
    pass
