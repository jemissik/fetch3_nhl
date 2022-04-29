import numpy as np
import xarray as xr

def format_inputs(nc_out):
    H = nc_out["ds_all"].H
    trans_2d = nc_out["ds_canopy"].trans_2d * 10**3 # 10**3 to convert m to kg

    #Get aboveground z indexes to slice the H dataset
    zind_canopy = np.arange(len(nc_out['ds_all'].z) - len(nc_out['ds_canopy'].z),len(nc_out['ds_all'].z))

    H_above = H.isel(z=zind_canopy)

    return H_above, trans_2d

def calc_sap_storage(H, params):
    sapwood_area = params.sapwood_area # m2
    dz = params.dz
    Phi0x = params.Phi_0
    p = params.p

    #params.sat_xylem is in [m3 h2o/m3xylem]
    thetasat = params.sat_xylem * 1000 # thetasat[kg m-3]
    taper_top = params.taper_top

    #convert H from MPa to Pa
    H = H * 10**6

    nz = len(H.z)

    taper = np.linspace(1, taper_top,nz)

    sapwood_area_z = sapwood_area * taper

    theta = thetasat * ((Phi0x / (Phi0x - H)) ** p) * sapwood_area_z

    storage = (theta.rolling(z=2).mean() * dz).sum(dim='z', skipna=True) # kg

    return storage

def calc_sapflux(H, trans_2d, params):
    """
    Calculates sapflux and total aboveground water storage of the tree.

    Parameters
    ----------
    H : _type_
        _description_
    trans_2d : datarray
        transpiration in kg s-1 m-1stem
    params : _type_
        _description_

    Returns
    -------
    sapflux : array-like
        Tree-level sap flux [kg s-1]
    storage : array-like
        Total aboveground water storage [kg]

    """
    dt = 1800
    dz = params.dz

    storage = calc_sap_storage(H, params)
    storage.name = 'storage'

    trans_tot = (trans_2d * dz).sum(dim='z', skipna=True)

    # Change in storage
    delta_S = storage.pad(time=(1,0)).diff(dim='time') / dt
    delta_S.name = 'delta_S'

    sapflux = trans_tot + delta_S
    sapflux.name = 'sapflux'

    ds_sapflux = xr.merge([sapflux, storage, delta_S])

    return ds_sapflux