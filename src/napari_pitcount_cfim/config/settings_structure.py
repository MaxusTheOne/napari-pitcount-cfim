import os
from typing import Optional

from pydantic import BaseModel, Field

def get_default_output_folder() -> str:
    base = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    return os.path.join(base, "napari-pitcount-cfim", "output")

def get_default_input_folder() -> str:
    base = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    return os.path.join(base, "napari-pitcount-cfim", "input")

class DebugSettings(BaseModel):
    """
    Settings for debugging.
    """
    debug: bool = Field(default=False)
    verbosity_level: int = Field(default=1)

class AutomationSettings(BaseModel):
    """
        Settings specifying what steps to run automatically.
    """
    folder_prompt: bool = Field(default=True, description="Prompt for folder selection.")

class FileSettings(BaseModel):
    """
        Settings for file handling.
    """
    input_folder: str = Field(default_factory=get_default_input_folder, description="Folder containing the input files.")
    output_folder: str = Field(default_factory=get_default_output_folder, description="Folder to save the output files.")

    # Attempted virtual fields
    debug: Optional[bool] = Field(default=None, exclude=True)
    folder_prompt: Optional[bool] = Field(default=None, exclude=True)


class CellposeSettings(BaseModel):
    """
    Settings for the Cellpose segmentation.
    """
    diameter: Optional[float] = Field(default=None
                            , description="Diameter of the cells in pixels. If None, Cellpose will estimate it.")
    border_filter: bool = Field(default=True)
    model_type: str = Field(default="cyto3")
    gpu: bool = Field(default=False)
    sharpen_radius: int = Field(default=0)

    # Attempted virtual fields
    debug: Optional[bool] = Field(default=None, exclude=True)


class CFIMSettings(BaseModel):
    """
        Settings for the napari pitcount CFIM plugin.

        Update the version number here after a change.
    """
    __version__: str = "0.7.6"

    version: str = Field(default=__version__)
    automation_settings: AutomationSettings = AutomationSettings()
    file_settings: FileSettings = FileSettings()
    cellpose_settings: CellposeSettings = CellposeSettings()
    debug_settings: DebugSettings = DebugSettings()

    def model_post_init(self, _context):
        """
            Post init for inheritance that does not show up in yaml.
        """

        self.file_settings.debug = self.debug_settings.debug
        self.file_settings.folder_prompt = self.automation_settings.folder_prompt
        self.cellpose_settings.debug = self.debug_settings.debug

    def as_dict_with_virtuals(self) -> dict:
        d = self.model_dump()
        d["file_settings"]["debug"] = self.file_settings.debug
        d["file_settings"]["folder_prompt"] = self.file_settings.folder_prompt
        d["cellpose_settings"]["debug"] = self.cellpose_settings.debug
        return d
