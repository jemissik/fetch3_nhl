"""
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

from fetch3.scaling import trans2d_to_tree


def main(cfg, output_dir, data_dir, to_model_res=True, write_output=True):
    """
    Calculate NHL transpiration.

    Parameters
    ----------
    cfg : FETCH3 config object
        Model configuration
    output_dir : pathlib.Path
        Directory to write output files to. If output_dir is None, NHL output is not written.
    data_dir : pathlib.Path
        Directory containing input data
    to_model_res : bool, optional
        Whether or not to calculate and return NHL in the resolution needed to run in FETCH, by default True.

    Returns
    -------
    LAD : array
        LAD profile
    NHL transpiration : xarray.DataArray
        NHL transpiration. If to_model_res is True, NHL is returned in the resolution
        needed to run in FETCH, with units of [m3 H2O m-2crown m-1stem s-1]. If to_model_res is False,
        NHL is returned as tree-level transpiration, with units of [kg H2O s-1].
    """
    logger = logging.getLogger(__name__)

    start = time.time()

    # Read in LAD and met data
    met_data = prepare_ameriflux_data(data_dir / cfg.input_fname, cfg)
    LADnorm_df = pd.read_csv(data_dir / cfg.LAD_norm)

    logger.info("Calculating NHL...")

    ds, LAD, zen = calc_NHL_timesteps(cfg, met_data, LADnorm_df)

    # Apply NHl scaling factor
    ds["NHL_trans_sp_stem"] = ds.NHL_trans_sp_stem * cfg.scale_nhl  # [kg H2O s-1 m-1stem m-2crown]
    ds["NHL_trans_leaf"] = ds.NHL_trans_leaf * cfg.scale_nhl  # [kg H2O m-2leaf s-1]
    ds = ds.assign_coords(species=cfg.species)

    # Nighttime transpiration
    ds["NHL_trans_sp_stem"] = calc_nighttime_trans(ds.NHL_trans_sp_stem, met_data.PPFD_IN, cfg.mean_crown_area_sp)
    ds["NHL_trans_leaf"] = calc_nighttime_trans(ds.NHL_trans_leaf, met_data.PPFD_IN, cfg.mean_crown_area_sp)

    logger.info(f"NHL calculations finished in {time.time() - start} s")

    if write_output:
        # logger.info("Saving NHL output...")
        nhl_trans_tot = trans2d_to_tree(ds.NHL_trans_sp_stem, cfg.mean_crown_area_sp, cfg.dz) # kg h20 s-1
        nhl_trans_tot.attrs = dict(
            units="kg h20 s-1", description="NHL transpiration per tree"
        )
        nhl_trans_tot = nhl_trans_tot.assign_coords(species=cfg.species)
        write_outputs_netcdf(output_dir, ds, filename=f'nhl_2d_{cfg.species}.nc')
        write_outputs_netcdf(output_dir, nhl_trans_tot, filename=f'nhl_tree_{cfg.species}.nc')
        # write_outputs({"zenith": zen, "LAD": LAD}, output_dir)

    if not to_model_res:
        return nhl_trans_tot, LAD

    if to_model_res:
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
