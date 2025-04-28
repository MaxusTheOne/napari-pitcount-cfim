class ImageHandler:
    """
        A class to handle and manage transfer of images between the napari viewer and the plugin.
    """
    def __init__(self, napari_viewer, parent=None, settings=None):
        self.viewer = napari_viewer
        self.parent = parent
        self.settings = settings
        self.image_data = None
        self.image_layer = None