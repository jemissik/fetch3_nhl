"""
##################
Initial conditions
##################
Calculates initial conditions based on specified soil moisture and assuming hydrostatic conditions in the plant

Initial conditions in the soil layers
- initial soil moisture conditions [m3 m-3] for each soil layer are specified in the configuration file
- corresponding water potential [Pa] is calculated using the van genuchten equation

Initial conditions in the plant:
- potential at bottom of roots equals the soil potential at that depth
- potential at height z = potential at bottom of roots + rho*g*z, where z=0 is the bottom of the roots
"""

import numpy as np

from fetch3.model_config import ConfigParams


def calc_potential_vangenuchten(theta, theta_r, theta_s, alpha, m, n, rho, g):
    """
    Calculates water potential from soil moisture, using van Genuchten equation

    Parameters
    ----------
    theta : float or np.ndarray
        soil water content [m3 m-3]
    theta_r : float
        residual water content [m3 m-3]
    theta_s : float
        saturated water content [m3 m-3]
    alpha : float
        empirical van Genuchten parameter [m-1]
    m : float
        empirical van Genuchten parameter [unitless]
    n :  float
        empirical van Genuchten parameter [unitless]
    rho : float
        density of water [kg m-3]
    g : float
        gravitational constant [m s-2]

    Returns
    -------
    water_potential_Pa : float or np.ndarray
        water potential [Pa]

    """

    effective_saturation = (theta - theta_r) / (theta_s - theta_r)
    water_potential_m = -((((1 / effective_saturation) ** (1 / m) - 1) ** (1 / n)) / alpha)
    water_potential_Pa = water_potential_m * rho * g

    return water_potential_Pa  # [Pa]


def initial_conditions(cfg: ConfigParams, q_rain, zind):
    """
    Calculate initial water potential conditions

    Parameters
    ----------
    cfg : dataclass
        model configuration
    q_rain : np.ndarray
        array of rain data
    zind : dataclass
        z index dataclass

    Returns
    -------
    H_initial: np.ndarray
        initial values for water potential [Pa] over the concatenated z domain (soil, roots, xylem)
    Head_bottom_H: np.ndarray
        water potential [Pa] for the bottom boundary. size is len(number of timesteps)

    """

    # soil
    H_initial_soil = np.piecewise(
        zind.z_soil,
        [zind.z_soil <= cfg.parameters.clay_d, zind.z_soil > cfg.parameters.clay_d],
        [
            calc_potential_vangenuchten(
                cfg.parameters.initial_swc_clay,
                cfg.parameters.theta_R1,
                cfg.parameters.theta_S1,
                cfg.parameters.alpha_1,
                cfg.parameters.m_1,
                cfg.parameters.n_1,
                cfg.Rho,
                cfg.g,
            ),
            calc_potential_vangenuchten(
                cfg.parameters.initial_swc_sand,
                cfg.parameters.theta_R2,
                cfg.parameters.theta_S2,
                cfg.parameters.alpha_2,
                cfg.parameters.m_2,
                cfg.parameters.n_2,
                cfg.Rho,
                cfg.g,
            ),
        ],
    )

    # roots

    # z index where roots begin (round to get rid of floating point precision error so it matches the z array)
    z_root_start = np.round(cfg.parameters.Soil_depth - cfg.parameters.Root_depth, decimals=5)
    H_initial_root_bottom = H_initial_soil[zind.z_soil == z_root_start]
    H_initial_root = H_initial_root_bottom - (zind.z_root - z_root_start) * cfg.Rho * cfg.g

    # xylem
    H_initial_xylem = H_initial_root_bottom - (zind.z_upper - z_root_start) * cfg.Rho * cfg.g

    # concatenated array for z domain
    H_initial = np.concatenate((H_initial_soil, H_initial_root, H_initial_xylem))

    # calculate water potential for the bottom boundary condition
    Head_bottom_H = np.full(
        len(q_rain),
        calc_potential_vangenuchten(
            cfg.parameters.soil_moisture_bottom_boundary,
            cfg.parameters.theta_R1,
            cfg.parameters.theta_S1,
            cfg.parameters.alpha_1,
            cfg.parameters.m_1,
            cfg.parameters.n_1,
            cfg.Rho,
            cfg.g,
        ),
    )

    # set bottom boundary for initial condition
    if cfg.model_options.BottomBC == 0:
        H_initial[0] = Head_bottom_H[0]

    return H_initial, Head_bottom_H
