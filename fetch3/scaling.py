"""
Provides scaling functions to convert common parameters to model parameters,
and convenience functions for unit conversions of transpiration.

Config parameters that will be provided by the user:

- :math:`\mathrm{LAI_p^{(sp)} \ [m^2_{leaf (sp)} / m^2_{ground}]}`: species-specific plot-level LAI
- :math:`\mathrm{SD [trees \ hectare^{-1}]}`: stand density [# trees per hectare]
- DBH (stem diameter at breast height [cm])
- Sapwood depth [cm]
- :math:`\mathrm{A_c^{(sp)} [m^2]}`: Mean crown area (species-specific)

Calculated parameters:

- species-specific crown-level LAI (LAIc_sp) [m2 leaf (sp) / m2 ground]
- sapwood area [m2]
- xylem area index (Aind_x) : [m2 xylem m-2 ground]
"""

import numpy as np


def calc_LAIc_sp(LAIp_sp, mean_crown_area_sp, stand_density_sp):
    r"""
    Calculates the crown-level species-specific LAI (LAIc_sp [m2 leaf (sp) / m2 ground projection of crown]), using
    the plot-level species-specific LAI (LAIp_sp [m2 leaf (sp) / m2 ground in plot]), mean crown area of the species [m2],
    and the stand density of the species in the plot [# of trees of sp / hectare].

    .. math::

        LAI_c^{(sp)} = \frac{LAI_p^{(sp)}}{A_c \times 10^{-4} \times SD^{(sp)}}



    Parameters
    ----------
    LAIp_sp : float
        Plot-level species-specific LAI [m2 leaf (sp) / m2 ground in plot]
    mean_crown_area_sp : float
        Mean crown area of the species [m2]
    stand_density_sp : float
        Stand density of the species in the plot [# of trees of sp / hectare]

    Returns
    -------
    LAIc_sp : float
        Crown-level species-specific LAI [m2 leaf (sp) / m2 ground projection of crown]
    """
    LAIc_sp = LAIp_sp / (mean_crown_area_sp * 10**-4 * stand_density_sp)
    return LAIc_sp


def calc_xylem_cross_sectional_area(DBH_cm, active_xylem_depth_cm):
    """
    Calculate xylem cross-sectional area from DBH and active xylem depth

    Parameters
    ----------
    DBH : float or array
        Diameter at breast height [cm]
    active_xylem_depth : float or array
        Active xylem depth [cm]

    Returns
    -------
    xylem_cross_sectional_area : float or array
        Cross-sectional area of active xylem [m2]
    """
    # Convert DBH and active xylem depth to m
    DBH_m = DBH_cm / 100
    active_xylem_depth_m = active_xylem_depth_cm / 100

    # Calculate stem cross-sectional area
    stem_cross_sectional_area = np.pi * (DBH_m / 2) ** 2

    # Calculate cross-sectional area of inner portion of stem (that is not active xylem)
    inner_cross_sectional_area = np.pi * (DBH_m / 2 - active_xylem_depth_m) ** 2

    # Xylem cross-sectional area
    xylem_cross_sectional_area = stem_cross_sectional_area - inner_cross_sectional_area

    return xylem_cross_sectional_area


def calc_Aind_x(xylem_cross_sectional_area, mean_crown_area_sp):
    """
    Calculates the crown-level xylem area index [m2 xylem m-2 crown projection]

    Parameters
    ----------
    xylem_cross_sectional_area : float
        sapwood area [m2]
    mean_crown_area_sp : float
        Mean crown area of species (crown projection to ground) [m2]
    """
    return xylem_cross_sectional_area / mean_crown_area_sp


# [ m3H2O m-2ground s-1 m-1stem]
def convert_trans2d_to_cm3hr(trans_2d, crown_area, dz):
    # [ m3H2O m-2crown s-1 m-1stem]

    # convert from per ground to per tree -> [m3h20 s-1]
    trans = trans2d_to_tree(trans_2d, crown_area, dz)

    # Convert m3/s to cm3/hr
    trans = convert_trans_m3s_to_cm3hr(trans)

    return trans


def convert_trans_m3s_to_cm3hr(trans):
    return trans * (100**3) * 60 * 60


def integrate_trans2d(trans_2d, dz):
    trans = (trans_2d * dz).sum(dim="z")
    return trans


def trans2d_to_tree(trans_2d, crown_area, dz):
    # [m3h20 s-1]
    trans = integrate_trans2d(trans_2d, dz) * crown_area
    return trans

def convert_sapflux_cm3hr_to_m3s(sapflux):
    return sapflux / ((100**3) * 60 * 60)

def convert_sapflux_cm3hr_to_mm30min(sapflux_cm3hr):
    """
    Converts aggregated plot-scale sapfluxnet data (in units of m3h20 m-2ground hr-1 to mm 30min-1

    Parameters
    ----------
    sapflux_cm3hr : [cm3h2o hr-1 m-2 plot]

    Returns
    -------
    scaled sapflux in mm 30min-1
    """
    sapflux_mm30min = (sapflux_cm3hr / ((100**3) * 60 * 60)) * 1800 * 1000
    return sapflux_mm30min
