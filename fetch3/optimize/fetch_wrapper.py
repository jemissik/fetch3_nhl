"""
Optimization wrapper for FETCH3.

These functions provide the interface between the optimization tool and FETCH3
- Setting up optimization experiment
- Creating directories for model outputs of each iteration
- Writing model configuration files for each iteration
- Starting model runs for each iteration
- Reading model outputs and observation data for model evaluation
- Defines objective function for optimization, and other performance metrics of interest
- Defines how results of each iteration should be evaluated
"""

import atexit
import os
import subprocess
import logging

from pprint import pformat

logger = logging.getLogger(__name__)

import yaml
from ax import Trial
from boa import (
    BaseWrapper,
    get_trial_dir,
    make_trial_dir,
    load_jsonlike,
    BOAConfig
)

# Keep these names in this module so older BOA configs can still reference
# fetch_data_func values such as "get_model_sapflux".
from fetch3.results.compare import (
    get_model_obs_summary,
    get_model_nhl_trans,
    get_model_obs,
    get_model_plot_trans,
    get_model_sapflux,
    get_model_swc,
    normalize_model_obs,
)
from fetch3.scaling import scale_sapflux, scale_transpiration


class Fetch3Wrapper(BaseWrapper):
    _processes = []
    config_file_name = "config.yml"
    fetch_data_funcs = {get_model_sapflux.__name__: get_model_sapflux,
                        get_model_plot_trans.__name__: get_model_plot_trans,
                        get_model_swc.__name__: get_model_swc,
                        get_model_nhl_trans.__name__: get_model_nhl_trans,
                        get_model_obs.__name__: get_model_obs,
                        get_model_obs_summary.__name__: get_model_obs_summary,
                        }

    def __init__(self, *args, **kwargs):
        self._model_trees = {}
        print(args, kwargs)
        super().__init__(*args, **kwargs)

    def load_config(self, config_path, *args, **kwargs):
        """
        Load config takes a configuration path of either a JSON file or a YAML file and returns
        your configuration dictionary.

        Load_config will (unless overwritten in a subclass), do some basic "normalizations"
        to your configuration for convenience. See :func:`.normalize_config`
        for more information about how the normalization works and what config options you
        can control.

        This implementation offers a default implementation that should work for most JSON or YAML
        files, but can be overwritten in subclasses if need be.

        Parameters
        ----------
        config_path
            File path for the experiment configuration file

        Returns
        -------
        BOAConfig
            loaded_config
        """
        config = load_jsonlike(config_path)

        if "model_trees" in config:
            parameter_keys = [["groups", key] for key in config.get("groups", {}).keys()]
            parameter_keys.extend([["model_trees", tree] for tree in config["model_trees"].keys()])
            for model_tree, parameters in config["model_trees"].items():
                self._model_trees[model_tree] = parameters.pop("parents", None)
        elif "species_parameters" in config:
            parameter_keys = [["species_parameters", key] for key in config.get("species_parameters", {}).keys()]
            parameter_keys.append(["site_parameters"])
        else:
            raise ValueError("No model trees or species parameters found in config file")

        self.config = BOAConfig(parameter_keys=parameter_keys, **config)
        return self.config

    def write_configs(self, trial: Trial) -> None:
        """
        Write model configuration file for a trial (model run). This is the config file used by FETCH3
        for the model run.

        The config file is written as ```config.yml``` inside the trial directory.

        Parameters
        ----------
        trial: Trial
            The trial to deploy.

        Returns
        -------
        str
            Path for the config file
        """
        trial_dir = make_trial_dir(self.experiment_dir, trial.index)
        config_dict = self.config.boa_params_to_wpr(trial.arm.parameters, self.config.mapping)
        config_dict["model_options"] = self.model_settings

        logging.info(pformat(config_dict))

        if self._model_trees:
            for model_tree, parameters in config_dict["model_trees"].items():
                parameters["parents"] = self._model_trees[model_tree]

        with open(trial_dir / self.config_file_name, "w") as f:
            # Write model options from loaded config
            # Parameters for the trial from Ax
            yaml.dump(config_dict, f)
            return f.name

    def run_model(self, trial: Trial):

        trial_dir = get_trial_dir(self.experiment_dir, trial.index)
        config_path = trial_dir / self.config_file_name

        # model_dir = self.model_settings["model_dir"]

        # os.chdir(model_dir)

        cmd = self.script_options.run_model.format(config_path=config_path,
                                                    data_path=self.model_settings['data_path'],
                                                    trial_dir=trial_dir)

        args = cmd.split()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
        self._processes.append(popen)

    def set_trial_status(self, trial: Trial, log_file='fetch3.log') -> None:
        """ "Get status of the job by a given ID. For simplicity of the example,
        return an Ax `TrialStatus`.
        """
        log_file = get_trial_dir(self.experiment_dir, trial.index) / log_file

        if log_file.exists():
            with open(log_file, "r") as f:
                contents = f.read()
            if "Error completing Run! Reason:" in contents:
                trial.mark_failed()
            elif "run complete" in contents:
                trial.mark_completed()

    def fetch_trial_data(self, trial: Trial, metric_properties: dict, metric_name: str, *args, **kwargs):

        modelfile = (
            get_trial_dir(self.experiment_dir, trial.index) / metric_properties[metric_name]["output_fname"]
        )

        fetch_data_func = self.fetch_data_funcs[metric_properties[metric_name]["fetch_data_func"]]

        y_pred, y_true = fetch_data_func(
            modelfile,
            **metric_properties[metric_name]
        )
        return dict(y_pred=y_pred, y_true=y_true)


class NHLWrapper(Fetch3Wrapper):
    fetch_data_funcs = {get_model_nhl_trans.__name__: get_model_nhl_trans,
                       }
    def __init__(self, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)

    def run_model(self, trial: Trial):

        trial_dir = get_trial_dir(self.experiment_dir, trial.index)
        config_path = trial_dir / self.config_file_name

        # model_dir = self.model_settings["model_dir"]

        # os.chdir(model_dir)

        cmd = self.script_options.run_model.format(config_path=config_path,
                                                    data_path=self.model_settings['data_path'],
                                                    trial_dir=trial_dir,
                                                    # species=self.ex_settings['species']
                                                    )

        args = cmd.split()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
        self._processes.append(popen)

    def set_trial_status(self, trial: Trial, log_file='nhl.log') -> None:
        return super().set_trial_status(trial, log_file)


def exit_handler():
    for process in Fetch3Wrapper._processes:
        process.kill()


atexit.register(exit_handler)
