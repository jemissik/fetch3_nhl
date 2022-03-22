from ax.service.ax_client import AxClient

from fetch_wrapper import read_experiment_config, create_experiment_dir, create_trial_dir, write_configs, run_model, evaluate
from pathlib import Path
import shutil

def main():
    # Settings
    config_file= "/Users/jmissik/Desktop/repos/fetch3_nhl/optimize/opt_model_config.yml"

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
        run_model(ex_settings['model_path'], config_dir, trial_dir)

        modelfile = trial_dir / ex_settings['output_fname'] # model output

        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(modelfile, ex_settings['obsfile']))

    # Add code to save the experiment
    ax_client.save_to_json_file(filepath = Path(experiment_dir) / "ax_client_snapshot.json")

if __name__ == "__main__":
    main()