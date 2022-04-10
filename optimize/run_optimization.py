"""
FETCH3 optimization
-------------------
Runs optimization for FETCH3.
Options for the optimization are set in the .yml file.

This script is meant to be run from the command line, and the optimization configuration .yml
file is specified as a command line argument, for example::
      python run_optimization.py --config_path /Users/jmissik/Desktop/repos/fetch3_nhl/optimize/umbs_optimization_config.yml

See optimization_config_template.yml for an example configuration file.

See ``optimization_results.ipynb`` for an example of how to explore the optimization results. 
"""

from ax.service.ax_client import AxClient

from fetch_wrapper import read_experiment_config, create_experiment_dir, create_trial_dir, write_configs, run_model, evaluate
from pathlib import Path
import shutil

import argparse

# Default paths for config file, input data, and model output
parent_path = Path(__file__).parent
default_config_path = parent_path / 'optimization_config_template.yml'

# Taking command line arguments for path of config file, input data, and output directory
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', nargs='?', default=default_config_path)
    args = parser.parse_args()
    config_file = args.config_path
except SystemExit:
    #use default options if invalid command line arguments are given
    config_file = default_config_path

# config_file = "/Users/jmissik/Desktop/repos/fetch3_nhl/optimize/umbs_optimization_config.yml"

def main(config_file):

    # Set up the experiment
    ax_client = AxClient()  # Initialize Ax client for experiment
    params, ex_settings, model_settings = read_experiment_config(config_file)  # Read experiment config

    # Create the experiment
    ax_client.create_experiment(
        name=ex_settings['experiment_name'],
        parameters=params,
        objective_name=ex_settings['objective_name'],
        minimize=True,  # defaults to False.
    )

    # Set up experiment directory
    experiment_dir = create_experiment_dir(ex_settings['working_dir'], ax_client)

    # Copy the experiment config to the experiment directory
    shutil.copyfile(config_file, experiment_dir / Path(config_file).name)

    # Run the optimization
    for i in range(ex_settings['ntrials']):
        parameters, trial_index = ax_client.get_next_trial()

        # create directory for trial
        trial_dir = create_trial_dir(experiment_dir, trial_index)

        # write parameters to config file for model
        config_dir = write_configs(trial_dir, parameters, model_settings)

        #run model
        run_model(ex_settings['model_path'], config_dir, ex_settings['data_path'], trial_dir)

        modelfile = trial_dir / ex_settings['output_fname'] # model output

        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(modelfile, ex_settings['obsfile'], ex_settings, model_settings, parameters))

    # Add code to save the experiment
    ax_client.save_to_json_file(filepath = Path(experiment_dir) / "ax_client_snapshot.json")

if __name__ == "__main__":
    main(config_file)