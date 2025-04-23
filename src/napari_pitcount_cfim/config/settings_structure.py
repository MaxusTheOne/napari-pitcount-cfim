import os

from pydantic import BaseModel, Field


class PitcountSettings(BaseModel):
    """
    Settings for the napari pitcount CFIM plugin.
    """
    __version__: str = "0.1.0"

    debug: bool = Field(default=False)
    version: str = Field(default=__version__)
    output_folder: str = Field(default_factory=get_default_output_folder)


def get_default_output_folder() -> str:
    base = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    return os.path.join(base, "napari-pitcount-cfim", "output")