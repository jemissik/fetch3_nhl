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
from fetch3.results.plotting import plot_sap


from boa import scheduler_from_json_file
from ax.service.utils.report_utils import get_standard_plots, exp_to_df

import warnings

# Suppress FutureWarnings from Ax
warnings.filterwarnings('ignore', message='Passing literal json')
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')

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


def load_obs_data(filein, timevar):
    obsdf = pd.read_csv(filein, index_col=[timevar], parse_dates=[timevar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)  # Change to tz-naive time
    return obsdf


def calc_canopy1d(res):

    #TODO needs to be updated for multiple trees

    dz = res.cfg.model_options.dz
    crown_area = res.cfg.parameters.mean_crown_area_sp
    tree_name = res.cfg.species

    nhl = convert_trans2d_to_cm3hr(res.canopy.nhl_trans_2d, crown_area, dz).sel(species=tree_name)
    trans = convert_trans2d_to_cm3hr(res.canopy.trans_2d, crown_area, dz).sel(species=tree_name)
    canopy1d = xr.Dataset(dict(nhl=nhl, trans=trans))
    canopy1d.nhl.attrs = dict(
        units="cm3 hr-1", description=("NHL transpiration in cm3 hr-1")
    )
    canopy1d.trans.attrs = dict(
        units="cm3 hr-1", description=("transpiration in cm3 hr-1")
    )

    canopy1d['theta'] = res.canopy.theta.mean(dim="z", skipna=True)

    canopy1d['H'] = res.canopy.H.mean(dim='z', skipna=True)

    return canopy1d


def rename_vars(ds, label):
    renames = {}
    for var in ds.data_vars:
        renames[var] = f'{var}_{label}'
    return ds.rename(renames)


def calc_canopy_daily(res):

    # mean
    daily_mean = rename_vars(res.canopy1d.resample(time='1D').mean(), 'mean')

    # min
    daily_min = rename_vars(res.canopy1d.resample(time='1D').min(), 'min')

    # max
    daily_max = rename_vars(res.canopy1d.resample(time='1D').max(), 'max')

    # sum
    ds = res.canopy1d[['nhl', 'trans']]
    daily_tot = rename_vars(ds.resample(time='1D').sum(), 'tot')

    # concat
    daily = xr.merge([daily_mean, daily_min, daily_max, daily_tot])

    return daily

class OptResults:

    def __init__(self, output_dir):

        self.exp_dir = Path(output_dir)
        self.scheduler_fp = self.exp_dir / 'scheduler.json'
        self.scheduler = scheduler_from_json_file(self.scheduler_fp)
        self.experiment = self.scheduler.experiment
        self.exp_df = exp_to_df(self.experiment)
        self.best_trial = self.scheduler.best_raw_trials()
        self.best_trial_index = list(self.best_trial.keys())[0]
        self.dir_best_trial = self.exp_dir / str(self.best_trial_index).zfill(6)

        self.get_obs_dict_from_opt_cfg()
        self.plots = None
        try:
            self.get_opt_plots()
        except:
            print("Error loading plots")


    def get_obs_dict_from_opt_cfg(self):
        from fetch3.optimize.fetch_wrapper import Fetch3Wrapper

        wrapper = Fetch3Wrapper()
        opt_config_path = list(Path(self.exp_dir).glob('*.yml'))[0]
        wrapper.load_config(opt_config_path)
        metrics = wrapper.config.objective.metrics

        self.obs_dict = {}
        for metric in metrics:
            fname = Path(metrics[0].properties['obs_file']).name
            self.obs_dict[metric.name] = {'fname': fname,
                                    'tvar': metric.properties.get('obs_tvar', 'TIMESTAMP')
            }


    def get_wp50(self):
        params = self.best_trial[list(self.res.best_trial)[0]]['params']
        wp50_param = [x for x in list(params.keys()) if 'wp_s50' in x][0]
        wp50 =params[wp50_param]
        wp50_Mpa = wp50 * 10 ** -6
        return wp50_Mpa
    
    def get_opt_plots(self):
        self.plots = get_standard_plots(self.experiment, self.scheduler.generation_strategy.model)


    def show_plots(self):
        if self.plots:
            for plot in self.plots:
                plot.show()
        else:
            print("No plots found")


class Results:
    """
    Loading and plotting model results
    - get best trial from an optimization

    """

    def __init__(self, output_dir, opt=True, label=None, config_name=None, data_dir=None, obs=None):

        if opt:
            self.opt = OptResults(output_dir)
            self.output_dir = self.opt.dir_best_trial
        else:
            self.output_dir = Path(output_dir)

        # Set label for the model run
        if label:
            self.label=label
        else:
            # TODO check config for an experiment name first
            self.label = self.output_dir.name

        # Load config for the model run
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

        if obs is None:
            obs = {}
            obs['met'] =  {
                            'fname': self.cfg.model_options.input_fname,
                            'tvar': 'TIMESTAMP_START'
                            }
            if opt:
                obs.update(self.opt.obs_dict)

        self.load_model_results()
        self.load_obs(data_dir, obs)


    def load_obs(self, data_dir, obs):
        self.data_dir = Path(data_dir)
        self.obs = {}


        for k in obs.keys():
            self.obs[k] = load_obs_data(self.data_dir / obs[k]['fname'], timevar=obs[k]['tvar'])

    def load_model_results(self):
        try:
            # Load results
            self.canopy, self.soil, self.roots, self.sapflux = load_model_outputs(self.output_dir)

            # Reassign coordinates
            # TODO should be removed eventually
            if self.soil.z.min().values >= 0:  # old model output where 0 is bottom of soil column, not soil surface
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

            # daily canopy
            self.canopy_daily = calc_canopy_daily(self)

        except:
            print("Error loading outputs")
            # self.canopy = None
            # self.soil = None
            # self.roots = None
            # self.sapflux = None


    def plot_sap(self, obs_df, obs_var=None, **kwargs):
        if obs_var is None:
            obs_var = self.cfg.species
        return plot_sap(self.canopy1d, self.obs[obs_df], obs_var=obs_var, **kwargs).opts(xlim=(self.start_time, self.end_time))


class MultiResults:

    def __init__(self, results):

        data_vars = ['soil', 'roots', 'canopy', 'sapflux']

        for var in data_vars:
            self.__setattr__(var, concat_ds(results, var))

        self.experiments = [result.label for result in results]
