import argparse
import faulthandler
import os
import subprocess

import napari

def launch_napari_dev_mode(mode='dev'):
    """
        Launch Napari in dev mode:


        Current issues:
        - dev mode messes with napari opening. Causing incorrect window size while the program still thinks it's full screen. Only visual.
        - Opening napari console with shortcut removes exposed variables. Makes the debug class unusable.
            - Works fine when opened with the console button.
    """
    print(f"Launching Napari in {mode} mode...")

    viewer = napari.Viewer(ndisplay=2, show= False)

    import cellpose #Imports to expose to the Ipython console
    # Activate your plugin (psf_analysis_CFIM)
    try:
        viewer.window.add_plugin_dock_widget("napari-pitcount-cfim", widget_name="Analyze pit count - CFIM")
        print("Activated plugin 'psf-analysis-CFIM'.")
    except ValueError:
        print("Plugin 'psf-analysis-CFIM' not found or failed to load.")
    if os.getenv('PITCOUNT_CFIM_NO_GUI', "0") == "0":
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
    parser.add_argument(
        "--no-gui", action="store_true", help="Run pipeline without GUI. Useful for testing or automation."
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, default=0, help="Set verbosity level (default: 0, max: 2)"
    )
    parser.add_argument(
        "--input-folder", "-i", type=str, help="Input folder (only with --no-gui)"
    )
    args = parser.parse_args()

    # enforce dependency
    if args.input_folder and not args.no_gui:
        parser.error("--input-folder can only be used when --no-gui is set")

    if args.no_gui:
        print("Running in non-GUI mode. This will disable the graphical interface and run the pipeline in the background.")

        # Setup non-GUI environment variables
        if args.input_folder:
            os.environ["PITCOUNT_CFIM_INPUT_FOLDER"] = args.input_folder
        if args.verbosity is not None:
            os.environ["PITCOUNT_CFIM_VERBOSITY"] = str(args.verbosity)
    os.environ["PITCOUNT_CFIM_NO_GUI"] = "1" if args.no_gui else "0"


    # Launch the appropriate mode
    if args.dev or args.no_gui:
        # Run the custom dev mode setup
        launch_napari_dev_mode(mode='dev' if args.dev else 'no-gui')
    else:
        launch_napari()