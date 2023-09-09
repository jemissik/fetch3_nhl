"""
Functions for calculating sap storage and sap flux from the model outputs
"""

import numpy as np
import xarray as xr

from fetch3.model_config import ConfigParams
from fetch3.scaling import integrate_trans2d


def format_inputs(canopy_ds, crown_area):
    """
    Formats variables for sapflux calculation

    Parameters
    ----------
    canopy_ds : xarray.Dataset
        Canopy dataset containing:
            - trans_2d: Transpiration[ m3H2O m-2crown_projection s-1 m-1stem]
            - H: Water potential [MPa]
    crown_area : float
        Crown area [m2]

    Returns
    -------
    H_above: xarray.DataArray
        Canopy water potential [MPa]
    trans_2d_tree: xarray.DataArray
        Transpiration [m3H2O s-1 m-1stem]
    """

    # trans_2d [ m3H2O m-2crown_projection s-1 m-1stem]
    # multiply by crown area to get transpiration in [m3 s-1 m-1stem]
    trans_2d_tree = canopy_ds.trans_2d * crown_area
    nhl_2d_tree = canopy_ds.nhl_trans_2d * crown_area

    H_above = canopy_ds.H

    return H_above, trans_2d_tree, nhl_2d_tree


def calc_xylem_theta(H_MPa, cfg: ConfigParams):
    """
    Calculates xylem water content based on water potential

    Parameters
    ----------
    H : xarray.DataArray
        Water potential [MPa]
    cfg : dataclass
        Model configuration parameters

    Returns
    -------
    theta: xarray.DataArray
        Xylem water content [m3 h2o/m3xylem]
    """
    Phi0x = cfg.parameters.Phi_0
    p = cfg.parameters.p

    # Convert H to Pa
    H = H_MPa * 10**6

    # cfg.sat_xylem is in [m3 h2o/m3xylem]
    thetasat = cfg.parameters.sat_xylem

    theta = thetasat * ((Phi0x / (Phi0x - H)) ** p)

    return theta


def calc_sap_storage(H_MPa, cfg: ConfigParams):
    """
    Calculates sap storage based on water potential

    Parameters
    ----------
    H : xarray.DataArray
        Water potential [MPa]
    cfg : dataclass
        Model configuration parameters

    Returns
    -------
    storage: xarray.DataArray
        Sap storage [m3]
    """
    sapwood_area = cfg.parameters.sapwood_area  # m2
    dz = cfg.model_options.dz
    taper_top = cfg.parameters.taper_top

    taper = np.linspace(1, taper_top, len(H_MPa.z))

    sapwood_area_z = sapwood_area * taper

    theta = calc_xylem_theta(H_MPa, cfg)  # m3 h2o / m3 xylem

    storage = (theta.rolling(z=2).mean() * sapwood_area_z * dz).sum(dim="z", skipna=True)  # m3
    theta = theta.mean(dim="z", skipna=True)

    return theta, storage


def calc_sapflux(canopy_ds, cfg: ConfigParams):
    """
    Calculates sapflux and total aboveground water storage of the tree.

    Parameters
    ----------
    H : xarray.DataArray
        Water potential [MPa]
    trans_2d_tree : xarray.DataArray
        transpiration [m3 s-1 m-1stem]
    cfg : dataclass
        Model configuration parameters

    Returns
    -------
    ds_sapflux : xarray.Dataset
        Dataset containing:
            - sapflux: Tree-level sap flux [m3 s-1]
            - storage: Total aboveground water storage [m3]
            - delta_s: Change in aboveground water storage from the previous timestep [m3]

    """
    dt = cfg.model_options.dt
    dz = cfg.model_options.dz
    crown_area = cfg.parameters.mean_crown_area_sp

    theta, storage = calc_sap_storage(canopy_ds.H, cfg)
    theta.name = "theta"
    storage.name = "storage"

    trans_2d_tree = canopy_ds.trans_2d * crown_area
    nhl_2d_tree = canopy_ds.nhl_trans_2d * crown_area

    trans_tot = integrate_trans2d(trans_2d_tree, dz)  # [m3 s-1]
    trans_tot.name = "trans"

    nhl_tot = integrate_trans2d(nhl_2d_tree, dz)  # [m3 s-1]
    nhl_tot.name = "nhl_trans"

    # Change in storage
    delta_S = storage.pad(time=(1, 0)).diff(dim="time") / dt
    delta_S.name = "delta_S"

    sapflux = trans_tot + delta_S
    sapflux.name = "sapflux"

    ds_sapflux = xr.merge([sapflux, storage, delta_S, theta, trans_tot, nhl_tot])

    # Add plot-level sapflux for the species
    ds_sapflux['sapflux_plot'] = (0.0001 * cfg.parameters.stand_density_sp) * ds_sapflux.sapflux
    ds_sapflux['storage_plot'] = (0.0001 * cfg.parameters.stand_density_sp) * ds_sapflux.storage
    ds_sapflux['delta_S_plot'] = (0.0001 * cfg.parameters.stand_density_sp) * ds_sapflux.delta_S



    # Add metadata to dataset
    ds_sapflux.sapflux.attrs = dict(units="m3 s-1", description="Tree-level sap flux")
    ds_sapflux.storage.attrs = dict(units="m3", description="Total aboveground water storage")
    ds_sapflux.theta.attrs = dict(units="m3 m-3", description="Xylem water content")
    ds_sapflux.trans.attrs = dict(units="m3 s-1", description="Tree-level transpiration")
    ds_sapflux.nhl_trans.attrs = dict(units="m3 s-1", description="Tree-level NHL transpiration")



    ds_sapflux.delta_S.attrs = dict(
        units="m3", description="Change in aboveground water storage from the previous timestep"
    )
    ds_sapflux.sapflux_plot.attrs = dict(
        units="m3 H2O m-2ground s-1", description=("Total plot-level sapflux for the species, calculated as tree-level "
                                                   "sap flux * (# of trees per m2)")
    )
    ds_sapflux.storage_plot.attrs = dict(
        units="m3 H2O m-2ground", description=("Total plot-level aboveground water storage for the species, calculated "
                                               "as tree-level water storage * (# of trees per m2)")
    )

    ds_sapflux.delta_S_plot.attrs = dict(
        units="m3 H2O m-2ground", description=("Change in plot-level aboveground water storage from the previous "
                                               "timestep, calculated as the change in tree-level water storage * (# of trees per m2)")
    )
    return ds_sapflux
