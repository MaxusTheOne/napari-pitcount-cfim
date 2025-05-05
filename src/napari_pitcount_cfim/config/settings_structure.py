import os
from pydantic import BaseModel, Field

def get_default_output_folder() -> str:
    base = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    return os.path.join(base, "napari-pitcount-cfim", "output")

class DebugSettings(BaseModel):
    """
    Settings for debugging.
    """
    debug: bool = Field(default=False)
    verbosity_level: int = Field(default=1)

class CellposeSettings(BaseModel):
    """
    Settings for the Cellpose segmentation.
    """
    border_filter: bool = Field(default=True)
    model_type: str = Field(default="cyto3")
    gpu: bool = Field(default=False)


class CFIMSettings(BaseModel):
    """
    Settings for the napari pitcount CFIM plugin.
    """
    __version__: str = "0.3.0"

    version: str = Field(default=__version__)
    output_folder: str = Field(default_factory=get_default_output_folder)
    input_folder: str = Field(default_factory=get_default_output_folder)
    cellpose_settings: CellposeSettings = CellposeSettings()
    debug_settings: DebugSettings = DebugSettings()


