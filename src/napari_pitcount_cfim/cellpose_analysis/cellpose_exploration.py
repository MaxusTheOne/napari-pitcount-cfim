#!/usr/bin/env python3
"""
Test script to run Cellpose segmentation on a TIFF file and visualize results in Napari.

Modify `TIFF_PATH` below to point to your .tiff file.
"""
import napari
from cellpose import models
import tifffile
import importlib.resources as pkg_resources

def cellpose_try(tiff_path):
    # Load the TIFF image (2D, 3D or 4D)
    img = tifffile.imread(tiff_path)
    viewer = napari.Viewer(ndisplay=2, show= False)

    # Initialize the Cellpose model (cytoplasm model by default)
    model = models.Cellpose(
        gpu=False,
        model_type='cyto3'
    )

    # Run segmentation
    masks, flows, styles, diams = model.eval(
        [img],                   # List of images
        channels=[1, 0],         # [cell_channel, nuclear_channel] (0-based)
        normalize=True,          # Intensity normalization
        invert=False,            # Do not invert intensities
        diameter=None,           # Use model diam_mean
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        augment=False,
        compute_masks=True
    )

    viewer.add_image(img, name='Original Image')
    viewer.add_labels(masks[0], name='Cell Masks')
    # Add flow channels (HSV flows and probability)
    viewer.add_image(flows[0][0], name='Flow X')
    viewer.add_image(flows[0][1], name='Flow Y')
    viewer.add_image(flows[0][2], name='Cell Probability')

    viewer.window.show()
    napari.run()
    # Print estimated diameters
    print(f"Estimated diameters: {diams}")


# Path to your TIFF file; modify as needed.
pkg_root = pkg_resources.files("src")

TIFF_PATH = pkg_root.joinpath("resources","test_files", "P06 1-Tile-16-1-channel.tiff")

if __name__ == '__main__':
    print(f"Dev |  Path: {TIFF_PATH}")
    cellpose_try(TIFF_PATH)
