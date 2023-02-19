"""
########
Runs NHL
########
This file runs the NHL module.
This version is intended to be called by FETCH3, and reads configs from the FETCH3
model config.

It returns NHL transpiration to be used by FETCH3, and also writes NHL
transpiration to a netcdf file.

If running NHL as a standalone module, use ``main_standalone.py``.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fetch3.met_data import prepare_ameriflux_data
from fetch3.nhl_transpiration.NHL_functions import *


def main(cfg, output_dir, data_dir):
    logger = logging.getLogger(__name__)

    start = time.time()

    ####################################################
    # The rest of the file should be the same as main.py
    ###################################################

    # Read in LAD and met data
    met_data = prepare_ameriflux_data(data_dir / cfg.input_fname, cfg)
    LADnorm_df = pd.read_csv(data_dir / cfg.LAD_norm)

    logger.info("Calculating NHL...")

    ds, LAD, zen = calc_NHL_timesteps(cfg, met_data, LADnorm_df)

    # NHL scaling
    ds["NHL_trans_sp_stem"] = ds.NHL_trans_sp_stem * cfg.scale_nhl
    ds["NHL_trans_leaf"] = ds.NHL_trans_leaf * cfg.scale_nhl

    # Nighttime transpiration
    ds["NHL_trans_sp_stem"] = calc_nighttime_trans(ds.NHL_trans_sp_stem, met_data.PPFD_IN, cfg.mean_crown_area_sp)
    ds["NHL_trans_leaf"] = calc_nighttime_trans(ds.NHL_trans_leaf, met_data.PPFD_IN, cfg.mean_crown_area_sp)

    logger.info(f"NHL calculations finished in {time.time() - start} s")

    # logger.info("Saving NHL output...")
    # write_outputs_netcdf(output_dir, ds)
    # write_outputs({"zenith": zen, "LAD": LAD}, output_dir)

    logger.info(f"Interpolating NHL to the time resolution for FETCH3...")
    # Interpolate to model time resolution
    # time in seconds
    ds2 = ds.assign_coords(
        {
            "time": pd.to_timedelta(
                pd.to_datetime(ds.time.values) - pd.to_datetime(ds.time.values[0])
            )
            / np.timedelta64(1, "s")
        }
    )

    # New time and space coordinates matching model resolution
    model_ts = np.arange(0, len(ds.time) * cfg.dt + cfg.dt0, cfg.dt0)
    model_z = np.arange(0, cfg.Hspec, cfg.dz)

    da = (
        ds2.NHL_trans_sp_stem * 10**-3
    )  # NHL in units of kg m-2crown m-1stem s-1 #* 10**-3 to convert kg to m

    NHL_modelres = da.interp(z=model_z, time=model_ts, assume_sorted=True, kwargs={"fill_value": 0})

    # logger.info("Saving NHL_modelres output...")
    # # write NHL output to netcdf
    # NHL_modelres.to_netcdf(output_dir / "nhl_modelres_trans_out.nc")
    NHL_modelres = NHL_modelres.data.transpose()

    logger.info(f"NHL module finished in {time.time() - start} s")
    return NHL_modelres, LAD
