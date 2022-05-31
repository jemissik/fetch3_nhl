"""
##################
Initial conditions
##################

Assume hydrostatic conditions in plant
- potential at bottom equals the potential specified in the soil layer where the root bottom is
- potential at height z = initial head at bottom or roots + rho*g*h

For soil layers
- specify an initial soil moisture for each soil layer
- try just using a constant potential over each layer (but check to see if this causes issues in the solver).
- might need to do some interpolation between layers
"""
import numpy as np


def initial_conditions(cfg, q_rain, zind):

    # clay layer

    # sand layer

    # roots
    # P0 = head at the soil layer where the root bottom is [Pa]
    # xylem

    dz = cfg.dz

    initial_H = np.zeros(shape=zind.nz)

    factor_soil = (cfg.H_init_soilbottom - (cfg.H_init_soilmid)) / (
        int((cfg.clay_d - cfg.cte_clay) / dz)
    )  # factor for interpolation

    # soil
    for i in np.arange(0, len(zind.z_soil), 1):
        if 0.0 <= zind.z_soil[i] <= cfg.cte_clay:
            initial_H[i] = cfg.H_init_soilbottom
        if cfg.cte_clay < zind.z_soil[i] <= zind.z[zind.nz_clay]:
            initial_H[i] = initial_H[i - 1] - factor_soil  # factor for interpolation
        if cfg.clay_d < zind.z_soil[i] <= zind.z[zind.nz_r - 1]:
            initial_H[i] = cfg.H_init_soilmid

    initial_H[zind.nz_s - 1] = cfg.H_init_soilmid

    factor_xylem = (cfg.H_init_canopytop - (cfg.H_init_soilbottom)) / (
        (zind.z[-1] - zind.z[zind.nz_s]) / dz
    )

    # roots and xylem
    initial_H[zind.nz_s] = cfg.H_init_soilbottom
    for i in np.arange(zind.nz_s + 1, zind.nz, 1):
        initial_H[i] = initial_H[i - 1] + factor_xylem  # meters

    # putting initial condition in Pascal
    H_initial = initial_H * cfg.g * cfg.Rho  # Pascals

    ###########################################################################
    # BOTTOM BOUNDARY CONDITION FOR THE SOIL
    # The model contains different options, therefore this variable is created but
    # only used if you choose a  Dirichlet BC
    ######################################################################
    soil_bottom = np.zeros(shape=len(q_rain))
    for i in np.arange(0, len(q_rain), 1):
        soil_bottom[i] = cfg.soil_moisture_bottom_boundary

    # clay - van genuchten
    Head_bottom = (
        (
            ((cfg.theta_R1 - cfg.theta_S1) / (cfg.theta_R1 - soil_bottom)) ** (1 / cfg.m_1) - 1
        )
        ** (1 / cfg.n_1)
    ) / cfg.alpha_1
    Head_bottom_H = -Head_bottom * cfg.g * cfg.Rho  # Pa

    # model starts the simulation at the BOTTOM of the soil
    Head_bottom_H = np.flipud(Head_bottom_H)

    ############## inital condition #######################
    # setting profile for initial condition
    if cfg.BottomBC == 0:
        H_initial[0] = Head_bottom_H[0]

    return H_initial, Head_bottom_H
