import json
from xml.etree import ElementTree as ET

from aicsimageio.readers import CziReader
import xmltodict

from napari_pitcount_cfim.library_workarounds.RangeDict import RangeDict

debug = True


def metadata_dump(czi_read_file, channels) -> list[dict]:
    """
    Dumps the metadata of a .czi file to a json file.
    """

    raw_xml = ET.tostring(czi_read_file.metadata, encoding="utf-8")
    ome_dict = xmltodict.parse(raw_xml)

    if debug:
        print("[*] Metadata dump:")
        print(json.dumps(ome_dict, indent=2))

    metadata = {}
    image_dict = ome_dict["ImageDocument"]["Metadata"]["Information"]["Image"]
    aq_mode_setup_dict = ome_dict["ImageDocument"]["Metadata"]["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"][
        "AcquisitionModeSetup"]
    # print(json.dumps(ome_dict["ImageDocument"]["Metadata"]["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"], indent=2))
    track_setup_list = \
    ome_dict["ImageDocument"]["Metadata"]["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["MultiTrackSetup"][
        "TrackSetup"]

    pixcount = [image_dict["SizeY"], image_dict["SizeX"]]

    yx_spacing = [float(aq_mode_setup_dict["ScalingX"]) * 1e6, float(aq_mode_setup_dict["ScalingY"]) * 1e6]

    metadata["size"] = pixcount
    metadata["scale"] = yx_spacing
    metadata["units"] = "micrometre"
    wavelengths = []

    for track_dict in track_setup_list:
        if track_dict["Attenuators"]["Attenuator"]["Wavelength"] is not None:
            wavelengths.append(track_dict["Attenuators"]["Attenuator"]["Wavelength"])

    channel_metadata_list = []

    for channel in range(channels):
        wavelength = round(float(image_dict["Dimensions"]["Channels"]["Channel"][channel].get("EmissionWavelength", 0)), 0)
        channel_metadata = {"metadata": {**metadata, "wavelength": wavelength}
                            , "scale": yx_spacing, "units": metadata["units"],
                            "colormap": wavelength_to_color[wavelength]}

        channel_metadata_list.append(channel_metadata)

    return channel_metadata_list


wavelength_to_color = RangeDict(
    [
        (0, 380, "Grey"),
        (380, 450, "Violet"),
        (450, 485, "Blue"),
        (485, 500, "Cyan"),
        (500, 565, "Green"),
        (565, 590, "Yellow"),
        (590, 625, "Orange"),
        (625, 740, "Red")])
if __name__ == "__main__":
    czifile = CziReader(r"C:\napari-pitcount-cfim\examples\P06 1-Tile-16.CZI")
    metadata = metadata_dump(czifile, 2)

    print(json.dumps(metadata, indent=2))
