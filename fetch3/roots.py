import numpy as np


def feddes_root_stress(theta, theta1, theta2):
    """
    Calculates Feddes root water uptake stress function, from equations S.73, S.74, and S.75 of Silva et al. 2022

    Parameters
    ----------
    theta: np.ndarray
        Volumetric soil water content [m3 m-3]
    theta1: np.ndarray
        Soil water content below which root water uptake ceases [m3 m-3]
    theta2: np.ndarray
        Soil water content below which root water uptake starts decreasing [m3 m-3]

    Returns
    -------
    stress_roots: np.ndarray
        Output of Feddes root water stress reduction function [unitless]
    """

    conditions = [theta <= theta1, (theta > theta1) & (theta <= theta2), theta > theta2]
    outputs = [0, (theta - theta1)/(theta2 - theta1), 1]
    stress_roots = np.select(conditions, outputs)

    return stress_roots
