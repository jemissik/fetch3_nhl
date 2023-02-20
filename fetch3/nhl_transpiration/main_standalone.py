"""
###############################
Runs NHL as a standalone module
###############################
This file runs NHL as a standalone module (without running FETCH3).

It returns NHL transpiration, and also writes NHL
transpiration to a netcdf file.

If running NHL inside FETCH3, use main.py in the NHL module instead.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nhl_config import config_file, data_dir, output_dir, read_configs
from NHL_functions import *

logger = logging.getLogger(__file__)

start = time.time()
cfg = read_configs(config_file)

####################################################
# The rest of the file should be the same as main.py
###################################################

# Read in LAD and met data
met_data = pd.read_csv(data_dir / cfg.input_fname, parse_dates=[0])
LAD_data = pd.read_csv(data_dir / cfg.LAD_norm)

met_data = met_data[
    (met_data.Timestamp >= pd.to_datetime(cfg.start_time))
    & (met_data.Timestamp <= pd.to_datetime(cfg.end_time))
].reset_index(drop=True)

# total_crown_area_sp = cfg.LAI * cfg.crown_scaling / cfg.sum_LAI_plot * cfg.plot_area
total_crown_area_sp = cfg.total_crown_area_sp

logger.info("Calculating NHL...")

ds, LAD, zen = calc_NHL_timesteps(
    cfg.dz,
    cfg.Hspec,
    cfg.Cd,
    met_data,
    cfg.Vcmax25,
    cfg.alpha_gs,
    cfg.alpha_p,
    cfg.LAI,
    cfg.plot_area,
    total_crown_area_sp,
    cfg.mean_crown_area_sp,
    LAD_data[cfg.species],
    LAD_data.z_h,
    cfg.latitude,
    cfg.longitude,
    time_offset=cfg.time_offset,
    zenith_method=cfg.zenith_method,
    x=cfg.x,
)
logger.info(f"NHL calculations finished in {time.time() - start} s")

logger.info("Saving NHL output...")
write_outputs_netcdf(output_dir, ds)
write_outputs({"zenith": zen, "LAD": LAD}, output_dir)


logger.info(f"Interpolating NHL to the time resolution for FETCH3...")
# Interpolate to model time resolution
# time in seconds
ds2 = ds.assign_coords(
    {
        "time": pd.to_timedelta(pd.to_datetime(ds.time.values) - pd.to_datetime(ds.time.values[0]))
        / np.timedelta64(1, "s")
    }
)

# New time and space coordinates matching model resolution
model_ts = np.arange(0, len(ds.time) * cfg.dt + cfg.dt0, cfg.dt0)
model_z = np.arange(0, cfg.Hspec, cfg.dz)

# NHL transpiration in units of kg H2O s-1 m-1stem
# da = ds2.NHL_trans_leaf #NHL in units of kg H2O m-2 s-1
da = ds2.NHL_trans_sp_stem * 10**-3  # NHL in units of kg m-1stem s-1 #* 10**-3 to convert kg to m

NHL_modelres = da.interp(z=model_z, time=model_ts, assume_sorted=True, kwargs={"fill_value": 0})

logger.info("Saving NHL_modelres output...")
# write NHL output to netcdf
NHL_modelres.to_netcdf(output_dir / "nhl_modelres_trans_out.nc")
NHL_modelres = NHL_modelres.data.transpose()

logger.info(f"NHL module finished in {time.time() - start} s")
