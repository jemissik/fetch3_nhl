"""
Functions for calculating sap storage and sap flux from the model outputs
"""

import numpy as np
import xarray as xr

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

    H_above = canopy_ds.H

    return H_above, trans_2d_tree


def calc_sap_storage(H_MPa, cfg):
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
    sapwood_area = cfg.sapwood_area  # m2
    dz = cfg.dz
    Phi0x = cfg.Phi_0
    p = cfg.p

    # Convert H to Pa
    H = H_MPa * 10**6

    # cfg.sat_xylem is in [m3 h2o/m3xylem]
    thetasat = cfg.sat_xylem
    taper_top = cfg.taper_top

    nz = len(H.z)

    taper = np.linspace(1, taper_top, nz)

    sapwood_area_z = sapwood_area * taper

    theta = thetasat * ((Phi0x / (Phi0x - H)) ** p) * sapwood_area_z

    storage = (theta.rolling(z=2).mean() * dz).sum(dim="z", skipna=True)  # m3

    return storage


def calc_sapflux(H, trans_2d_tree, cfg):
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
    dt = cfg.dt
    dz = cfg.dz

    storage = calc_sap_storage(H, cfg)
    storage.name = "storage"

    trans_tot = integrate_trans2d(trans_2d_tree, dz)  # [m3 s-1]

    # Change in storage
    delta_S = storage.pad(time=(1, 0)).diff(dim="time") / dt
    delta_S.name = "delta_S"

    sapflux = trans_tot + delta_S
    sapflux.name = "sapflux"

    ds_sapflux = xr.merge([sapflux, storage, delta_S])

    # Add plot-level sapflux for the species
    ds_sapflux['sapflux_plot'] = (0.0001 * cfg.stand_density_sp) * ds_sapflux.sapflux
    ds_sapflux['storage_plot'] = (0.0001 * cfg.stand_density_sp) * ds_sapflux.storage
    ds_sapflux['delta_S_plot'] = (0.0001 * cfg.stand_density_sp) * ds_sapflux.delta_S



    # Add metadata to dataset
    ds_sapflux.sapflux.attrs = dict(units="m3 s-1", description="Tree-level sap flux")
    ds_sapflux.storage.attrs = dict(units="m3", description="Total aboveground water storage")
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
