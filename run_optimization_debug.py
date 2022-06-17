"""
Convenience script for running the model in a debugger (i.e., when you don't want
to run the model from the command line).
Here, you can pass config_path, data_path, and output_path directly rather than as
command line arguments.
The paths here are for the default locations, but you can change these if you want to use
different files or directories.
"""

from pathlib import Path

from fetch3.optimize.run_optimization import run

# These paths point to the default locations
parent_path = Path(__file__).resolve().parent
# config_path = parent_path / "config_files" / "model_config.yml"

# If you want to use different files or directories, you can change these paths, for example:
config_path = Path("/Users/jmissik/Desktop/repos/fetch3_nhl/config_files/opt_umbs_M8.yml")

run(config_path)
