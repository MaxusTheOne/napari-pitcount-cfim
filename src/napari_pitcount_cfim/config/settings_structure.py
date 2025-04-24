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


class CFIMSettings(BaseModel):
    """
    Settings for the napari pitcount CFIM plugin.
    """
    __version__: str = "0.1.0"

    version: str = Field(default=__version__)
    output_folder: str = Field(default_factory=get_default_output_folder)
    debug_settings: DebugSettings = DebugSettings()


