"""
Convenience functions:
- loading model outputs
- plotting results
"""
import xarray as xr
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import yaml

from pathlib import Path
from fetch3.utils import load_yaml
from fetch3.optimize.fetch_wrapper import get_model_sapflux, get_model_swc
from fetch3.model_config import get_multi_config



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


def plot_2d_slice(ds=None, z=None, var=None):

    plot = ds[var].sel(z=z).hvplot.line(x='time')
    return plot

def plot_precip_and_swc(model_ds, z=None, obs=None, obs_P='P_F', obs_swc='SWC_F_MDS_1', scale_obs_swc=True):

    obs_swc = obs[obs_swc]

    # If obs are in % instead of fraction
    if scale_obs_swc:
        obs_swc = obs_swc / 100

    precip_plot = obs[obs_P].hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_obs_plot = obs_swc.hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_model_plot = model_ds.THETA.sel(z=z).hvplot.line(x='time', label='model SWC')
    swc_plot = (swc_obs_plot * swc_model_plot).opts(ylim=(0,0.3))
    return (precip_plot + swc_plot).cols(1)

class Results:

    def __init__(self, output_dir, label=None, config_name=None, data_dir=None):

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
        except:
            print("Error loading outputs")
            # self.canopy = None
            # self.soil = None
            # self.roots = None
            # self.sapflux = None

        # Load met data

        # Load sapflux observation data

class MultiResults:

    def __init__(self, results):

        data_vars = ['soil', 'roots', 'canopy', 'sapflux']

        for var in data_vars:
            self.__setattr__(var, concat_ds(results, var))

        self.experiments = [result.label for result in results]
