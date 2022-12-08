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

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from ax import Trial
from boa import (
    BaseWrapper,
    get_trial_dir,
    make_trial_dir,
    normalize_config,
    boa_params_to_wpr, load_jsonlike
)

from fetch3.scaling import convert_trans_m3s_to_cm3hr, convert_sapflux_m3s_to_mm30min


def get_model_plot_trans(modelfile, obs_file, obs_var, output_var, **kwargs):
    # Read in observation data - partitioned fluxnet data
    timestamp_col = 'TIMESTAMP_START'
    obsdf = pd.read_csv(obs_file, parse_dates=[timestamp_col])
    obsdf = obsdf.set_index(timestamp_col)

    # Read in model output
    modeldf = xr.load_dataset(modelfile)
    modeldf = modeldf.sel(species=output_var)

    # # Slice met data to just the time period that was modeled
    obsdf = obsdf.loc[modeldf.time.data[0]: modeldf.time.data[-1]]

    # # Convert model output to the same units as the input data
    # # tower data is in mm 30min-1
    modeldf["sapflux_plot_mm30min"] = convert_sapflux_m3s_to_mm30min(modeldf.sapflux_plot)

    # # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]
    modeldf = modeldf.sapflux_plot_mm30min.isel(time=np.arange(1, len(modeldf.time) - 1))

    not_nans = ~obsdf[obs_var].isna()
    obsdf_not_nans = obsdf[obs_var].loc[not_nans]
    modeldf_not_nans = modeldf.data[not_nans]

    return modeldf_not_nans, obsdf_not_nans


def get_model_sapflux(modelfile, obs_file, obs_var, output_var, **kwargs):
    """
    Read in observation data model output for a trial, which will be used for
    calculating the objective function for the trial.

    Parameters
    ----------
    modelfile : str
        File path to the model output
    obs_file : str
        File path to the observation data
    model_settings: dict
        dictionary with model settings read from model config file

    Returns
    -------
    model_output: pandas Series
        Model output
    obs: pandas Series
        Observations

    ..todo::
        * Add options to specify certain variables from the observation/output files
        * Add option to read from .nc file

    """
    # Read config file

    # Read in observation data
    obsdf = pd.read_csv(obs_file, parse_dates=[0])
    # Converting time since sapfluxnet data is in GMT
    obsdf["Timestamp"] = obsdf.TIMESTAMP.dt.tz_convert("EST").dt.tz_localize(None)
    obsdf = obsdf.set_index("Timestamp")


    # Read in model output
    modeldf = xr.load_dataset(modelfile)
    modeldf = modeldf.sel(species=output_var)

    # Slice met data to just the time period that was modeled
    obsdf = obsdf.loc[modeldf.time.data[0] : modeldf.time.data[-1]]

    # Convert model output to the same units as the input data
    # Sapfluxnet data is in cm3 hr-1
    modeldf["sapflux_scaled"] = convert_trans_m3s_to_cm3hr(modeldf.sapflux)

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]
    modeldf = modeldf.sapflux_scaled.isel(time=np.arange(1, len(modeldf.time) - 1))

    not_nans = ~obsdf[obs_var].isna()
    obsdf_not_nans = obsdf[obs_var].loc[not_nans]
    modeldf_not_nans = modeldf.data[not_nans]

    return modeldf_not_nans, obsdf_not_nans

def get_model_swc(modelfile, obs_file, obs_var, output_var, species, **kwargs):
    """
    Read in observation data model output for a trial, which will be used for
    calculating the objective function for the trial.

    Parameters
    ----------
    modelfile : str
        File path to the model output
    obs_file : str
        File path to the observation data
    model_settings: dict
        dictionary with model settings read from model config file

    Returns
    -------
    model_output: pandas Series
        Model output
    obs: pandas Series
        Observations

    ..todo::
        * Add options to specify certain variables from the observation/output files
        * Add option to read from .nc file

    """
    # Read config file

    # Read in observation data
    obsdf = pd.read_csv(obs_file, parse_dates=[0])
    obsdf["Timestamp"] = obsdf.TIMESTAMP_START
    obsdf = obsdf.set_index("Timestamp")

    # Read in model output
    modeldf = xr.load_dataset(modelfile)
    modeldf = modeldf.sel(z=5.9, species=species) * 100  #TODO

    # Slice met data to just the time period that was modeled
    obsdf = obsdf.loc[modeldf.time.data[0] : modeldf.time.data[-1]]

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]
    modeldf = modeldf[output_var].isel(time=np.arange(1, len(modeldf.time) - 1))

    not_nans = ~obsdf[obs_var].isna()
    obsdf_not_nans = obsdf[obs_var].loc[not_nans]
    modeldf_not_nans = modeldf.isel(time=not_nans).data.transpose()

    return modeldf_not_nans, obsdf_not_nans


def scale_sapflux(sapflux, dz, mean_crown_area_sp, total_crown_area_sp, plot_area):
    """Scales sapflux from FETCH output (in kg s-1) to W m-2"""
    scaled_sapflux = sapflux * 2440000 / mean_crown_area_sp * total_crown_area_sp / plot_area
    return scaled_sapflux


def scale_transpiration(trans, dz, mean_crown_area_sp, total_crown_area_sp, plot_area):
    """Scales transpiration from FETCH output (in m H20 m-2crown m-1stem s-1) to W m-2"""
    scaled_trans = (trans * 1000 * dz * 2440000 * total_crown_area_sp / plot_area).sum(
        dim="z", skipna=True
    )
    return scaled_trans


class Fetch3Wrapper(BaseWrapper):
    _processes = []
    config_file_name = "config.yml"
    fetch_data_funcs = {get_model_sapflux.__name__: get_model_sapflux,
                        get_model_plot_trans.__name__: get_model_plot_trans,
                        get_model_swc.__name__: get_model_swc}

    def load_config(self, config_path, *args, **kwargs) -> dict:
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
        dict
            loaded_config
        """
        config = load_jsonlike(config_path, normalize=False)
        parameter_keys = [["species_parameters", key] for key in config.get("species_parameters", {}).keys()]
        parameter_keys.append(["site_parameters"])
        self.config = normalize_config(config=config, parameter_keys=parameter_keys)
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
        config_dict = boa_params_to_wpr(trial.arm.parameters, self.config["optimization_options"]["mapping"])
        config_dict["model_options"] = self.model_settings

        logging.info(pformat(config_dict))

        with open(trial_dir / self.config_file_name, "w") as f:
            # Write model options from loaded config
            # Parameters for the trial from Ax
            yaml.dump(config_dict, f)
            return f.name

    def run_model(self, trial: Trial):

        trial_dir = get_trial_dir(self.experiment_dir, trial.index)
        config_path = trial_dir / self.config_file_name

        model_dir = self.ex_settings["model_dir"]

        os.chdir(model_dir)

        cmd = self.script_options["run_cmd"].format(config_path=config_path,
                                                    data_path=self.ex_settings['data_path'],
                                                    trial_dir=trial_dir)

        args = cmd.split()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
        self._processes.append(popen)

    def set_trial_status(self, trial: Trial) -> None:
        """ "Get status of the job by a given ID. For simplicity of the example,
        return an Ax `TrialStatus`.
        """
        log_file = get_trial_dir(self.experiment_dir, trial.index) / "fetch3.log"

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


def exit_handler():
    for process in Fetch3Wrapper._processes:
        process.kill()


atexit.register(exit_handler)
