"""
Provides scaling functions to convert common parameters to model parameters

Config parameters that will be provided by the user:

- :math:`\mathrm{LAI_p^{(sp)} \ [m^2_{leaf (sp)} / m^2_{ground}]}`: species-specific plot-level LAI
- :math:`SD [trees \ hectare^{-1}]`: stand density [# trees per hectare]
- DBH (stem diameter at breast height #TODO units [m or cm])
- Active xylem fraction (or depth?) #TODO[m or cm]
- Crown Area (Ac_sp)

Need to calculate:

- species-specific crown-level LAI (LAIc_sp) [m2 leaf (sp) / m2 ground]
- Stem area / xylem area
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

def calc_xylem_cross_sectional_area(DBH, active_xylem_depth):
    """
    Calculate xylem cross-sectional area from DBH and active xylem depth

    Parameters
    ----------
    DBH : float
        Diameter at breast height [cm]
    active_xylem_depth : float
        Active xylem depth [cm]

    Returns
    -------
    xylem_cross_sectional_area : float
        Cross-sectional area of active xylem [cm2]
    """
    # Calculate stem cross-sectional area
    stem_cross_sectional_area = np.pi * (DBH/2) ** 2

    # Calculate cross-sectional area of inner portion of stem below sapwood
    inner_cross_sectional_area = np.pi * (DBH/2 - active_xylem_depth) ** 2

    # Xylem cross-sectional area
    xylem_cross_sectional_area = stem_cross_sectional_area - inner_cross_sectional_area

    return xylem_cross_sectional_area