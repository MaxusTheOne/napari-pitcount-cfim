import argparse
import faulthandler
import subprocess

import napari

def launch_napari_dev_mode(czi_file=None, points=None):
    """
        Launch Napari in dev mode:


        Current issues:
        - dev mode messes with napari opening. Causing incorrect window size while the program still thinks it's full screen. Only visual.
        - Opening napari console with shortcut removes exposed variables. Makes the debug class unusable.
            - Works fine when opened with the console button.
    """
    print("Launching Napari in dev mode...")

    viewer = napari.Viewer(ndisplay=2, show= False)

    import cellpose #Imports to expose to the Ipython console
    # Activate your plugin (psf_analysis_CFIM)
    try:
        viewer.window.add_plugin_dock_widget("napari-pitcount-cfim", widget_name="Analyze pit count - CFIM")
        print("Activated plugin 'psf-analysis-CFIM'.")
    except ValueError:
        print("Plugin 'psf-analysis-CFIM' not found or failed to load.")

    viewer.window.show()
    napari.run()


def launch_napari():
    """Launch Napari."""
    try:
        print("Launching Napari...")
        subprocess.check_call(['napari'])
    except FileNotFoundError:
        print("Napari is not installed. Please install it by running 'pip install napari[all]'")
        exit(1)


if __name__ == "__main__":
    faulthandler.enable()
    # Setup argparse for handling dev mode
    parser = argparse.ArgumentParser(description="Start the Napari plugin with optional dev mode. Expects the plugin to be installed.")
    parser.add_argument(
        "--dev", action="store_true", help="Launch Napari in 'dev mode' for testing purposes."
    )


    args = parser.parse_args()

    # Launch the appropriate mode
    if args.dev:
        # Run the custom dev mode setup
        launch_napari_dev_mode()
    else:
        launch_napari()