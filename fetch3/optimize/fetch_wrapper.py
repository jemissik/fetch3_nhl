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
    load_jsonlike,
    BOAConfig
)

from fetch3.scaling import convert_trans_m3s_to_cm3hr, convert_sapflux_m3s_to_mm30min


def get_model_plot_trans(modelfile, obs_file, obs_var, output_var, obs_tvar='TIMESTAMP_START', **kwargs):

    # Read in observation data
    obsdf = pd.read_csv(obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time

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


def get_model_sapflux(modelfile, obs_file, obs_var, output_var, obs_tvar='TIMESTAMP', hour_range=None, normalize=True, **kwargs):
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
        * Add option to read from .nc file

    """

    # Read in observation data
    obsdf = pd.read_csv(obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time

    # Read in model output
    modelds = xr.load_dataset(modelfile)
    if 'species' in modelds.dims:
        modelds = modelds.sel(species=output_var)

    # Convert model output to the same units as the input data
    # Sapfluxnet data is in cm3 hr-1
    modelds["sapflux_scaled"] = convert_trans_m3s_to_cm3hr(modelds.sapflux)

    modeldf = modelds.squeeze(drop=True).to_dataframe()

    # Merge model and obs dataframes
    df = pd.merge(modeldf, obsdf[[obs_var]], how='left', right_index=True, left_index=True, suffixes=['model', 'obs'])

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]

    # Drop rows with NaN
    df = df.dropna()

    if normalize:
        df['sapflux_scaled'] = (df['sapflux_scaled'] - df['sapflux_scaled'].mean()) / df['sapflux_scaled'].std()
        df[obs_var] = (df[obs_var] - df[obs_var].mean()) / df[obs_var].std()

    if hour_range:
        df = df[(df.index.hour >= hour_range[0]) & (df.index.hour <= hour_range[1])]

    return df['sapflux_scaled'], df[obs_var]

def get_model_nhl_trans(modelfile, obs_file, obs_var, output_var, hour_range=None, scaling_factor=None, obs_tvar='TIMESTAMP', **kwargs):
    # Read in observation data
    obsdf = pd.read_csv(obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time

    # Read in model output
    modelds = xr.load_dataset(modelfile)
    if 'species' in modelds.dims:
        modelds = modelds.sel(species=output_var)


    # Convert model output to the same units as the input data
    # Sapfluxnet data is in cm3 hr-1
    # 1d NHL output is in kg h20 s-1
    modelds["nhl_scaled"] = convert_trans_m3s_to_cm3hr(modelds.NHL_trans_sp_stem * 10**-3) #* 10**-3 to convert kg to m3

    modeldf = modelds.squeeze(drop=True).to_dataframe()

    # Merge model and obs dataframes
    df = pd.merge(modeldf, obsdf[[obs_var]], how='left', right_index=True, left_index=True, suffixes=['model', 'obs'])

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]

    # Drop rows with NaN
    df = df.dropna()

    if scaling_factor:
        df[obs_var] = df[obs_var] * scaling_factor

    if hour_range:
        df = df[(df.index.hour >= hour_range[0]) & (df.index.hour <= hour_range[1])]

    return df['nhl_scaled'], df[obs_var]

def get_model_swc(modelfile, obs_file, obs_var, output_var, species, obs_tvar='TIMESTAMP_START', percent_units=True, obs_depth=0.1, **kwargs):
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

    # Read in observation data
    obsdf = pd.read_csv(obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time

    if percent_units:
        obsdf[obs_var] = obsdf[obs_var] / 100

    # Read in model output
    modeldf = xr.load_dataset(modelfile)

    # find depth
    z_soil_surface = modeldf.z.max().values
    z_sel = z_soil_surface - obs_depth

    modeldf = modeldf.sel(z=z_sel, species=species, method='nearest')

    # Slice met data to just the time period that was modeled
    obsdf = obsdf.loc[modeldf.time.data[0] : modeldf.time.data[-1]]

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]
    modeldf = modeldf[output_var].isel(time=np.arange(1, len(modeldf.time) - 1))

    not_nans = ~obsdf[obs_var].isna()
    obsdf_not_nans = obsdf[obs_var].loc[not_nans]
    modeldf_not_nans = modeldf.isel(time=not_nans).data.transpose()

    return modeldf_not_nans, obsdf_not_nans


def get_model_obs(modelfile, obs_file, obs_var, output_var, species, obs_tvar='TIMESTAMP_START', obs_multiplier=True, obs_z=None, **kwargs):
    """
    Read in observation data and model output for a trial. This function can be used for 1d and 2d model outputs
    where observations only need a scalar multiplier to convert to the same units as the model output. For 2d model
    outputs, the observations are compared to the z-slice of the model output that is closest to the observation
    height/depth.

    Parameters
    ----------
    modelfile : str
        File path to the model output file
    obs_file : str
        File path to the observation data file
    obs_var : str
        Column name of the observation variable
    output_var : str
        Name of the model output variable
    species : str
        Species
    obs_tvar : str, optional
        Name of the time column in the observation data, by default 'TIMESTAMP'
    obs_multiplier : float, optional
        Scalar multiplier to apply to the observation data in order to convert units to the model output. If `None`, no
        multiplier is applied.
    obs_z : float, optional
        Depth/height [m] of the observation data, where 0 is the soil surface. Aboveground is positive, belowground is
        negative. If `None`, the depth is set to the max z in the model output (i.e. soil surface for soil outputs,
        canopy top for canopy outputs). If the model output is 1d, this parameter is ignored.

    Returns
    -------
    array_like
        model data
    array_like
        observation data

    """

    # Read in observation data
    obsdf = pd.read_csv(obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time

    if obs_multiplier:
        obsdf[obs_var] = obsdf[obs_var] * obs_multiplier

    # Read in model output
    modelds = xr.load_dataset(modelfile)

    # Select species
    if 'species' in modelds.dims:
        modelds = modelds.sel(species=species)

    # Get depth slice for 2d outputs
    if 'z' in modelds.dims:
        # select depth
        if obs_z is None:
            z_sel = modelds.z.values.max()
        else:
            z_sel = obs_z

        modelds = modelds.sel(z=z_sel, method='nearest')

    # Slice met data to just the time period that was modeled
    obsdf = obsdf.loc[modelds.time.data[0] : modelds.time.data[-1]]

    # remove first and last timestamp
    obsdf = obsdf.iloc[1:-1]
    modelds = modelds[output_var].isel(time=np.arange(1, len(modelds.time) - 1))

    not_nans = ~obsdf[obs_var].isna()
    obsdf_not_nans = obsdf[obs_var].loc[not_nans]
    modelds_not_nans = modelds.isel(time=not_nans).data.transpose()

    return modelds_not_nans, obsdf_not_nans


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
                        get_model_swc.__name__: get_model_swc,
                        get_model_nhl_trans.__name__: get_model_nhl_trans,
                        get_model_obs.__name__: get_model_obs,
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
