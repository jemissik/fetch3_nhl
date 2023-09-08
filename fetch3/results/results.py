"""
Convenience functions:
- loading model outputs
- plotting results
"""
import xarray as xr
import pandas as pd
import yaml

from pathlib import Path
from fetch3.utils import load_yaml
from fetch3.optimize.fetch_wrapper import get_model_sapflux, get_model_swc
from fetch3.model_config import get_multi_config
from fetch3.sapflux import calc_xylem_theta
from fetch3.scaling import convert_trans2d_to_cm3hr

from boa import scheduler_from_json_file


def clip_timerange(df, tmin=None, tmax=None):
    if tmin is None:
        tmin = df.index.min()

    if tmax is None:
        tmax = df.index.max()

    mask = (df.index >= tmin) & (df.index <= tmax)

    return df.loc[mask]


def load_model_outputs(model_output_path):

    filein = Path(model_output_path) / "ds_canopy.nc"
    canopy = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_soil.nc"
    soil = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_root.nc"
    roots = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_sapflux.nc"
    sapflux = xr.load_dataset(filein)

    return canopy, soil, roots, sapflux


def concat_ds(results, var):
    ds_list = [result.__getattribute__(var).expand_dims(experiment=[result.label]) for result in results]
    ds = xr.concat(ds_list, dim='experiment')
    return ds

def load_met_data(filein, timevar='TIMESTAMP_START'):
    met = met = pd.read_csv(filein, parse_dates=[timevar])
    met = met.set_index(timevar)
    return met


def calc_canopy1d(res):

    dz = res.cfg.model_options.dz
    crown_area = res.cfg.parameters.mean_crown_area_sp
    tree_name = res.cfg.species

    nhl = convert_trans2d_to_cm3hr(res.canopy.nhl_trans_2d, crown_area, dz).sel(species=tree_name)
    trans = convert_trans2d_to_cm3hr(res.canopy.trans_2d, crown_area, dz).sel(species=tree_name)
    canopy1d = xr.Dataset(dict(nhl= nhl, trans=trans))
    canopy1d.nhl.attrs = dict(
        units="cm3 hr-1", description=("NHL transpiration in cm3 hr-1")
    )
    canopy1d.trans.attrs = dict(
        units="cm3 hr-1", description=("transpiration in cm3 hr-1")
    )

    canopy1d['theta'] = res.canopy.theta.mean(dim="z", skipna=True)

    return canopy1d


def get_best_opt_results(exp_dir):
    exp_dir = Path(exp_dir)
    scheduler_fp = exp_dir / 'scheduler.json'
    scheduler = scheduler_from_json_file(scheduler_fp)
    best_trial = scheduler.best_raw_trials()
    best_trial_index = list(best_trial.keys())[0]
    dir_best_trial = exp_dir / str(best_trial_index).zfill(6)
    return best_trial, dir_best_trial


class Results:

    def __init__(self, output_dir, opt=True, label=None, config_name=None, data_dir=None, obs_file=None, obs_tvar='TIMESTAMP'):

        if opt:
            self.best_trial, self.dir_best_trial = get_best_opt_results(output_dir)
            self.output_dir = self.dir_best_trial
        else:
            self.output_dir = Path(output_dir)

        try:
            if config_name is None:
                files = self.output_dir.glob('*.yml')
                config_name = [f.name for f in files if f.name != 'calculated_params.yml'][0]

            print(config_name)
            self.config_path = self.output_dir / config_name

            # TODO assumes only one tree for now
            self.cfg = get_multi_config(self.config_path)[0]

        except:
            print("Error loading config")
            self.config = None

        if label:
            self.label=label
        else:
            # TODO check config for an experiment name first
            self.label = self.output_dir.name

        try:
            # Load results
            self.canopy, self.soil, self.roots, self.sapflux = load_model_outputs(self.output_dir)

            # Reassign coordinates
            self.canopy = self.canopy.assign_coords(z=self.canopy.z - self.cfg.parameters.Soil_depth)
            self.soil = self.soil.assign_coords(z=self.soil.z - self.cfg.parameters.Soil_depth)
            self.roots = self.roots.assign_coords(z=self.roots.z - self.cfg.parameters.Soil_depth)

            # Start and end times
            self.start_time = self.canopy.time.min().values
            self.end_time = self.canopy.time.max().values

            # Add stem water content
            self.canopy['theta'] = calc_xylem_theta(self.canopy['H'], self.cfg)

            #1d canopy
            self.canopy1d = calc_canopy1d(self)

        except:
            print("Error loading outputs")
            # self.canopy = None
            # self.soil = None
            # self.roots = None
            # self.sapflux = None

        data_dir = Path(data_dir)
        # Load met data
        if obs_file is None:
            self.obs_file = data_dir / self.cfg.model_options.input_fname
        else:
            self.obs_file = data_dir / obs_file

        self.obs = pd.read_csv(self.obs_file, index_col=[obs_tvar], parse_dates=[obs_tvar])
        # Load sapflux observation data

class MultiResults:

    def __init__(self, results):

        data_vars = ['soil', 'roots', 'canopy', 'sapflux']

        for var in data_vars:
            self.__setattr__(var, concat_ds(results, var))

        self.experiments = [result.label for result in results]
