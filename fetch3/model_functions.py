"""
###############
Model functions
###############

Core functions of FETCH3
Porous media flow, Picard iteration, etc
"""

import logging

import numpy as np
import pandas as pd
import torch
import xarray as xr
from numpy.linalg import multi_dot

from fetch3.model_config import ConfigParams, TranspirationScheme
from fetch3.roots import calc_root_K, feddes_root_stress, verma_root_mass_dist

logger = logging.getLogger(__name__)

# function for stem xylem: K and C
def Porous_media_xylem(arg, ap, bp, kmax, Aind_x, p, sat_xylem, Phi_0, z, nz_r, nz):

    # arg= potential [Pa]
    cavitation_xylem = np.zeros(shape=(len(arg)))

    for i in np.arange(0, len(cavitation_xylem), 1):
        if arg[i] > 0:
            cavitation_xylem[i] = 1
        else:
            cavitation_xylem[i] = 1 - 1 / (1 + np.exp(ap * (arg[i] - bp)))

    # Index Ax/As - area of xylem per area of soil
    # kmax = m/s
    K = kmax * Aind_x * cavitation_xylem

    # CAPACITANCE FUNCTION AS IN BOHRER ET AL 2005
    C = np.zeros(shape=len(z[nz_r:nz]))

    C = ((Aind_x * p * sat_xylem) / (Phi_0)) * ((Phi_0 - arg) / Phi_0) ** (-(p + 1))

    return C, K, cavitation_xylem


########################################################################################

# function for root xylem: K and C
def Porous_media_root(arg, ap, bp, Ksax, Aind_r, p, sat_xylem, Phi_0, nz_r, nz_s):
    # arg= potential (Pa)
    stress_kr = np.zeros(shape=(len(arg)))

    for i in np.arange(0, len(stress_kr), 1):
        if arg[i] > 0:
            stress_kr[i] = 1
        else:
            stress_kr[i] = 1 - 1 / (
                1 + np.exp(ap * (arg[i] - bp))
            )  # CAVITATION CURVE FOR THE ROOT XYLEM

    # Index Ar/As - area of root xylem per area of soil
    # considered 1 following VERMA ET AL 2014 {for this case}

    # Keax = effective root axial conductivity
    K = Ksax * Aind_r * stress_kr  # [m2/s Pa]

    # KEEPING CAPACITANCE CONSTANT - using value according to VERMA ET AL., 2014
    C = np.zeros(shape=nz_r - nz_s)
    # C[:]=1.1*10**(-11)  #(1/Pa)

    # CAPACITANCE FUNCTION AS IN BOHRER ET AL 2005

    # considering axial area rollowing basal area [cylinder]
    C = ((Aind_r * p * sat_xylem) / (Phi_0)) * ((Phi_0 - arg) / Phi_0) ** (-(p + 1))

    return C, K, stress_kr


###############################################################################

# vanGenuchten for soil K and C
def vanGenuchten(
    arg,
    z,
    g,
    Rho,
    clay_d,
    theta_S1,
    theta_R1,
    alpha_1,
    n_1,
    m_1,
    Ksat_1,
    theta_S2,
    theta_R2,
    alpha_2,
    n_2,
    m_2,
    Ksat_2,
    dtO,
):

    # arg = potential from Pascal to meters
    theta = np.zeros(shape=len(arg))
    arg = (arg) / (g * Rho)  # m

    Se = np.zeros(shape=len(arg))
    K = np.zeros(shape=len(arg))
    C = np.zeros(shape=len(arg))
    # considering l = 0.5

    for i in np.arange(0, len(arg), 1):
        if z[i] <= clay_d:  # clay_d=4.2m for verma

            if arg[i] < 0:
                # Compute the volumetric moisture content
                theta[i] = (theta_S1 - theta_R1) / (
                    (1 + (alpha_1 * abs(arg[i])) ** n_1) ** m_1
                ) + theta_R1  # m3/m3
                # Compute the effective saturation
                Se[i] = (theta[i] - theta_R1) / (theta_S1 - theta_R1)  ## Unitless factor
                # Compute the hydraulic conductivity
                K[i] = (
                    Ksat_1 * Se[i] ** (1 / 2) * (1 - (1 - Se[i] ** (1 / m_1)) ** m_1) ** 2
                )  # van genuchten Eq.8 (m/s) #
            if arg[i] >= 0:
                theta[i] = theta_S1
                K[i] = Ksat_1

            C[i] = (
                ((-alpha_1 * np.sign(arg[i]) * m_1 * (theta_S1 - theta_R1)) / (1 - m_1))
                * Se[i] ** (1 / m_1)
                * (1 - Se[i] ** (1 / m_1)) ** m_1
            )

        if z[i] > clay_d:  # sand

            if arg[i] < 0:
                # Compute the volumetric moisture content
                theta[i] = (theta_S2 - theta_R2) / (
                    (1 + (alpha_2 * abs(arg[i])) ** n_2) ** m_2
                ) + theta_R2  # m3/m3
                # Compute the effective saturation
                Se[i] = (theta[i] - theta_R2) / (theta_S2 - theta_R2)  ## Unitless factor
                # Compute the hydraulic conductivity
                K[i] = (
                    Ksat_2 * Se[i] ** (1 / 2) * (1 - (1 - Se[i] ** (1 / m_2)) ** m_2) ** 2
                )  # van genuchten Eq.8 (m/s) #
            if arg[i] >= 0:
                theta[i] = theta_S2
                K[i] = Ksat_2

            C[i] = (
                ((-alpha_2 * np.sign(arg[i]) * m_2 * (theta_S2 - theta_R2)) / (1 - m_2))
                * Se[i] ** (1 / m_2)
                * (1 - Se[i] ** (1 / m_2)) ** m_2
            )

    K = K / (Rho * g)  # since H is in Pa
    C = C / (Rho * g)  # since H is in Pa

    return C, K, theta, Se


###############################################################################

###############################################################################


def Picard(cfg: ConfigParams, H_initial, Head_bottom_H, zind, met, t_num, nt, output_dir, data_dir):
    # picard iteration solver, as described in the supplementary material
    # solution following Celia et al., 1990
    z_soil = zind.z_soil
    nz_s = zind.nz_s
    nz_r = zind.nz_r
    z_upper = zind.z_upper
    z = zind.z
    nz = zind.nz
    z_Above = zind.z_Above
    nz_sand = zind.nz_sand
    nz_clay = zind.nz_clay

    q_rain = met.q_rain
    NET_2d = met.NET_2d
    SW_in_2d = met.SW_in_2d
    delta_2d = met.delta_2d
    VPD_2d = met.VPD_2d
    Ta_2d = met.Ta_2d

    # Imports for PM transpiration
    if cfg.transpiration_scheme == TranspirationScheme.PM:
        from fetch3.canopy import calc_LAD
        from fetch3.pm_transpiration import (
            calc_pm_transpiration,
            jarvis_fd,
            jarvis_fleaf,
            jarvis_fs,
            jarvis_fTa,
        )

        LAD = calc_LAD(z_Above, cfg.model_options.dz, cfg.parameters.z_m, cfg.parameters.Hspec, cfg.parameters.L_m)

        # 2D stomata reduction functions and variables for canopy-distributed transpiration
        f_Ta_2d = jarvis_fTa(Ta_2d, cfg.parameters.kt, cfg.parameters.Topt)
        f_d_2d = jarvis_fd(VPD_2d, cfg.parameters.kd)
        f_s_2d = jarvis_fs(SW_in_2d, cfg.parameters.kr)

    # Imports for NHL transpiration
    elif cfg.transpiration_scheme == TranspirationScheme.NHL:
        import fetch3.nhl_transpiration.main as nhl
        from fetch3.nhl_transpiration.NHL_functions import (
            calc_stem_wp_response,
            calc_transpiration_nhl,
        )

        NHL_modelres, LAD = nhl.main(cfg, output_dir, data_dir)

    # Stem water potential [Pa]

    ######################################################################

    # Define matrices that we’ll need in solution(similar to celia et al.[1990])
    x = np.ones(((nz - 1), 1))
    DeltaPlus = np.diagflat(-np.ones((nz, 1))) + np.diagflat(x, 1)  # delta (i+1) -delta(i)

    y = -np.ones(((nz - 1, 1)))
    DeltaMinus = np.diagflat(np.ones((nz, 1))) + np.diagflat(y, -1)  # delta (i-1) - delta(i)

    p = np.ones(((nz - 1, 1)))
    MPlus = np.diagflat(np.ones((nz, 1))) + np.diagflat(p, 1)

    w = np.ones((nz - 1, 1))
    MMinus = np.diagflat(np.ones((nz, 1))) + np.diagflat(w, -1)

    ############################Initializing the pressure heads/variables ###################
    # only saving variables EVERY HALF HOUR
    dim = np.mod(t_num, 1800) == 0
    dim = sum(bool(x) for x in dim)

    H = np.zeros(shape=(nz, dim))  # Water potential [Pa]
    trans_2d = np.zeros(shape=(len(z_upper), dim))
    nhl_trans_2d = np.zeros(shape=(len(z_upper), dim))
    K = np.zeros(shape=(nz, dim))
    Capac = np.zeros(shape=(nz, dim))
    S_kx = np.zeros(shape=(nz - nz_r, dim))
    S_kr = np.zeros(shape=(nz_r - nz_s, dim))
    S_sink = np.zeros(shape=(nz_r - nz_s, dim))
    Kr_sink = np.zeros(shape=(nz_r - nz_s, dim))
    THETA = np.zeros(shape=(nz_s, dim))
    EVsink_ts = np.zeros(shape=((nz_r - nz_s), dim))
    infiltration = np.zeros(shape=dim)

    Pt_2d = np.zeros(shape=(len(z_upper), nt))

    S_stomata = np.zeros(shape=(len(z[nz_r:nz]), nt))
    S_S = np.zeros(shape=(nz, nt))
    theta = np.zeros(shape=(nz_s))
    Se = np.zeros(shape=(nz_s, nt))

    # H_initial = inital water potential [Pa]
    H[:, 0] = H_initial[:]

    # root mass distribution following VERMA ET AL 2O14
    r_dist = verma_root_mass_dist(cfg)

    ####################################################################################################################################

    # INITIALIZING THESE VARIABLES FOR ITERATIONS
    cnp1m = np.zeros(shape=(nz))
    knp1m = np.zeros(shape=(nz))
    stress_kx = np.zeros(shape=(nz - nz_r))
    stress_kr = np.zeros(shape=(nz_r - nz_s))
    deltam = np.zeros(shape=(nz))

    # vector for adding potentials in B matrix
    TS = np.zeros(shape=(nz_r))

    # Define an iteration counter
    niter = 0
    sav = 0

    for i in np.arange(1, nt, 1):
        # use nt for entire period

        # Initialize the Picard iteration solver - saving variables every half-hour
        if i == 1:
            hn = H[:, 0]  # condition for initial conditions
        else:
            hn = hnp1mp1  # condition for remaining time steps

        hnp1m = hn

        # Define a dummy stopping variable
        stop_flag = 0

        while stop_flag == 0:
            # =========================== above-ground xylem ========================
            # Get C,K,for soil, roots, stem

            # VanGenuchten relationships applied for the soil nodes
            cnp1m[0:nz_s], knp1m[0:nz_s], theta[:], Se[:, i] = vanGenuchten(
                hnp1m[0:nz_s],
                z_soil,
                cfg.g,
                cfg.Rho,
                cfg.parameters.clay_d,
                cfg.parameters.theta_S1,
                cfg.parameters.theta_R1,
                cfg.parameters.alpha_1,
                cfg.parameters.n_1,
                cfg.parameters.m_1,
                cfg.parameters.Ksat_1,
                cfg.parameters.theta_S2,
                cfg.parameters.theta_R2,
                cfg.parameters.alpha_2,
                cfg.parameters.n_2,
                cfg.parameters.m_2,
                cfg.parameters.Ksat_2,
                cfg.model_options.dt0,
            )

            # Equations for C, K for the root nodes
            cnp1m[nz_s:nz_r], knp1m[nz_s:nz_r], stress_kr[:] = Porous_media_root(
                hnp1m[nz_s:nz_r],
                cfg.parameters.ap,
                cfg.parameters.bp,
                cfg.parameters.Ksax,
                cfg.parameters.Aind_r,
                cfg.parameters.p,
                cfg.parameters.sat_xylem,
                cfg.parameters.Phi_0,
                nz_r,
                nz_s,
            )

            # Equations for C, K for stem nodes
            cnp1m[nz_r:nz], knp1m[nz_r:nz], stress_kx[:] = Porous_media_xylem(
                hnp1m[nz_r:nz],
                cfg.parameters.ap,
                cfg.parameters.bp,
                cfg.parameters.kmax,
                cfg.parameters.Aind_x,
                cfg.parameters.p,
                cfg.parameters.sat_xylem,
                cfg.parameters.Phi_0,
                z,
                nz_r,
                nz,
            )

            # % Compute the individual elements of the A matrix for LHS

            C = np.diagflat(cnp1m)

            # interlayer hydraulic conductivity - transition between roots and stem
            # calculated as a simple average
            knp1m[nz_r] = (knp1m[nz_r - 1] + knp1m[nz_r]) / 2

            # interlayer between clay and sand
            knp1m[nz_clay] = (knp1m[nz_clay] + knp1m[nz_clay + 1]) / 2

            # equation S.17
            kbarplus = (1 / 2) * np.matmul(MPlus, knp1m)  # 1/2 (K_{i} + K_{i+1})

            kbarplus[nz - 1] = 0  # boundary condition at the top of the tree : no-flux
            kbarplus[nz_s - 1] = 0  # boundary condition at the top of the soil

            Kbarplus = np.diagflat(kbarplus)

            # equation S.16
            kbarminus = (1 / 2) * np.matmul(MMinus, knp1m)  # 1/2 (K_{i-1} - K_{i})

            kbarminus[0] = 0  # boundary contition at the bottom of the soil
            kbarminus[nz_s] = 0  # boundary contition at the bottom of the roots : no-flux

            Kbarminus = np.diagflat(kbarminus)

            ##########ROOT WATER UPTAKE TERM ###########################

            stress_roots = feddes_root_stress(
                theta[nz_s - len(zind.z_root) : nz_s],
                zind.theta_1[nz_s - len(zind.z_root) : nz_s],
                zind.theta_2[nz_s - len(zind.z_root) : nz_s],
            )

            Kr = calc_root_K(r_dist, stress_roots, cfg)

            #######################################################################
            # tridiagonal matrix
            # LA note:
            # This matrix is invertable because kbarplus and kbarminus should never be 0 at the same place (under reasonable situatisn)
            # and then with deltaplus and deltaminus you atridiagonal matrix with values defined on every diagonal
            A = (1 / cfg.model_options.dt0) * C - (1 / (cfg.model_options.dz**2)) * (
                np.dot(Kbarplus, DeltaPlus) - np.dot(Kbarminus, DeltaMinus)
            )

            # Infiltration calculation - only infitrates if top soil layer is not saturated
            # equation S.53
            if cfg.model_options.UpperBC == 0:
                q_inf = min(q_rain[i], ((cfg.parameters.theta_S2 - theta[-1]) * (cfg.model_options.dz / cfg.model_options.dt0)))  # m/s

            ################################## SINK/SOURCE TERM ON THE SAME TIMESTEP #####################################
            # equation S.22 suplementary material
            if cfg.parameters.Root_depth == cfg.parameters.Soil_depth:
                # diagonals
                for k, e in zip(np.arange(0, nz_s, 1), np.arange(0, (nz_r - nz_s), 1)):
                    A[k, k] = A[k, k] - Kr[e]  # soil ---  from 0:soil top
                for j, w in zip(np.arange(nz_s, nz_r, 1), np.arange(0, (nz_r - nz_s), 1)):
                    A[j, j] = A[j, j] - Kr[w]  # root ---- from soil bottom :root top

                # terms outside diagonals
                for k, j, e in zip(
                    np.arange((nz_r - nz_s), nz_r, 1),
                    np.arange((0), nz_s, 1),
                    np.arange(0, (nz_r - nz_s), 1),
                ):
                    A[j, k] = +Kr[e]  # root

                for k, j, e in zip(
                    np.arange(nz_s, nz_r, 1), np.arange(0, nz_s, 1), np.arange(0, (nz_r - nz_s), 1)
                ):
                    A[k, j] = +Kr[e]  # soil

                # residual for vector Right hand side vector
                TS[0:nz_s] = -Kr * (hnp1m[0:nz_s] - hnp1m[nz_s:nz_r])  # soil
                TS[(nz_s):nz_r] = +Kr * (hnp1m[0:nz_s] - hnp1m[nz_s:nz_r])  # root

            else:
                # diagonals
                for k, e in zip(
                    np.arange(nz_s - (nz_r - nz_s), nz_s, 1), np.arange(0, (nz_r - nz_s), 1)
                ):
                    A[k, k] = A[k, k] - Kr[e]  # soil --- (soil-roots):soil
                for j, w in zip(np.arange(nz_s, nz_r, 1), np.arange(0, (nz_r - nz_s), 1)):
                    A[j, j] = A[j, j] - Kr[w]  # root ---- soil:root

                # terms outside diagonals
                for k, j, e in zip(
                    np.arange(nz_s, nz_r, 1),
                    np.arange(nz_s - (nz_r - nz_s), nz_s, 1),
                    np.arange(0, (nz_r - nz_s), 1),
                ):
                    A[j, k] = +Kr[e]  # root

                for k, j, e in zip(
                    np.arange(nz_s, nz_r, 1),
                    np.arange(nz_s - (nz_r - nz_s), nz_s, 1),
                    np.arange(0, (nz_r - nz_s), 1),
                ):
                    A[k, j] = +Kr[e]  # soil

                # residual for vector Right hand side vector
                TS[nz_s - (nz_r - nz_s) : nz_s] = -Kr * (
                    hnp1m[nz_s - (nz_r - nz_s) : nz_s] - hnp1m[nz_s:nz_r]
                )  # soil
                TS[(nz_s):nz_r] = +Kr * (
                    hnp1m[nz_s - (nz_r - nz_s) : nz_s] - hnp1m[nz_s:nz_r]
                )  # root

            ########################################################################################################

            ##########TRANSPIRATION FORMULATION #################

            # For PM transpiration
            if cfg.transpiration_scheme == TranspirationScheme.PM:  # 0: PM transpiration scheme
                Pt_2d[:, i] = calc_pm_transpiration(
                    SW_in_2d[:, i],
                    NET_2d[:, i],
                    delta_2d[i],
                    cfg.parameters.Cp,
                    VPD_2d[:, i],
                    cfg.parameters.lamb,
                    cfg.parameters.gama,
                    cfg.parameters.gb,
                    cfg.parameters.ga,
                    cfg.parameters.gsmax,
                    cfg.parameters.Emax,
                    f_Ta_2d[:, i],
                    f_s_2d[:, i],
                    f_d_2d[:, i],
                    jarvis_fleaf(hn[nz_r:nz], cfg.parameters.hx50, cfg.parameters.nl),
                    LAD,
                )
            # For NHL transpiration
            elif cfg.transpiration_scheme == 1:  # 1: NHL transpiration scheme
                Pt_2d[:, i] = calc_transpiration_nhl(
                    NHL_modelres[:, i],
                    calc_stem_wp_response(hn[nz_r:nz], cfg.parameters.wp_s50, cfg.parameters.c3).transpose(),
                )

            # SINK/SOURCE ARRAY : concatenating all sinks and sources in a vector
            S_S[:, i] = np.concatenate((TS, -Pt_2d[:, i]))  # vector with sink and sources

            # dummy variable to help breaking the multiplication into parts
            matrix2 = multi_dot([Kbarplus, DeltaPlus, hnp1m]) - multi_dot(
                [Kbarminus, DeltaMinus, hnp1m]
            )

            # % Compute the residual of MPFD (right hand side)

            R_MPFD = (
                (1 / (cfg.model_options.dz**2)) * (matrix2)
                + (1 / cfg.model_options.dz) * cfg.Rho * cfg.g * (kbarplus - kbarminus)
                - (1 / cfg.model_options.dt0) * np.dot((hnp1m - hn), C)
                + (S_S[:, i])
            )

            # bottom boundary condition - known potential - \delta\Phi=0
            if cfg.model_options.BottomBC == 0:
                A[1, 0] = 0
                A[0, 1] = 0
                A[0, 0] = 1
                R_MPFD[0] = 0

            if cfg.model_options.UpperBC == 0:  # adding the infiltration on the most superficial soil layer [1/s]
                R_MPFD[nz_s - 1] = R_MPFD[nz_s - 1] + (q_inf) / cfg.model_options.dz

            if cfg.model_options.BottomBC == 2:  # free drainage condition: F1-1/2 = K at the bottom of the soil
                R_MPFD[0] = R_MPFD[0] - (kbarplus[0] * cfg.Rho * cfg.g) / cfg.model_options.dz

            # Compute deltam for iteration level m+1 : equations S.25 to S.41 (matrix)
            # deltam = np.dot(linalg.pinv2(A),R_MPFD)
            A_ = torch.from_numpy(A)
            R_MPFD_ = torch.from_numpy(R_MPFD)
            deltam = torch.linalg.lstsq(A_, R_MPFD_, rcond=-1).solution.numpy()

            if np.max(np.abs(deltam[:])) < cfg.model_options.stop_tol:  # equation S.42
                stop_flag = 1
                hnp1mp1 = hnp1m + deltam

                # Bottom boundary condition at bottom of the soil
                # setting for the next time step value for next cycle
                if cfg.model_options.BottomBC == 0:
                    hnp1mp1[0] = Head_bottom_H[i]

                # saving output variables only every 30min
                if np.mod(t_num[i], 1800) == 0:
                    sav = sav + 1

                    H[:, sav] = hnp1mp1  # saving potential
                    trans_2d[:, sav] = Pt_2d[:, i]  # 1/s

                    if cfg.transpiration_scheme == TranspirationScheme.NHL:
                        nhl_trans_2d[:, sav] = NHL_modelres[:, i]

                    hsoil = hnp1mp1[nz_s - (nz_r - nz_s) : nz_s]
                    hroot = hnp1mp1[(nz_s):(nz_r)]
                    EVsink_ts[:, sav] = -Kr[:] * (hsoil - hroot)  # sink term soil #saving

                    # saving output variables
                    K[:, sav] = knp1m
                    THETA[:, sav] = theta
                    Capac[:, sav] = cnp1m
                    S_kx[:, sav] = stress_kx
                    S_kr[:, sav] = stress_kr
                    S_sink[:, sav] = stress_roots
                    Kr_sink[:, sav] = Kr

                    if cfg.model_options.UpperBC == 0 and q_rain[i] > 0:
                        infiltration[sav] = q_inf
                niter = niter + 1

                if cfg.model_options.print_run_progress:
                    if (niter % cfg.model_options.print_freq) == 0:
                        logger.info("calculated time steps: %d" % niter)

            else:
                hnp1mp1 = hnp1m + deltam
                hnp1m = hnp1mp1

    return (
        H * (10 ** (-6)),
        K,
        S_stomata,
        theta,
        S_kx,
        S_kr,
        C,
        Kr_sink,
        Capac,
        S_sink,
        EVsink_ts,
        THETA,
        infiltration,
        trans_2d,
        nhl_trans_2d,
    )


# Calculating water balance from model outputs
def format_model_output(
    species,
    H,
    K,
    S_stomata,
    theta,
    S_kx,
    S_kr,
    C,
    Kr_sink,
    Capac,
    S_sink,
    EVsink_ts,
    THETA,
    infiltration,
    trans_2d,
    nhl_trans_2d,
    dt,
    start_time,
    end_time,
    dz,
    cfg: ConfigParams,
    zind,
):
    ####################### Water balance ###################################

    theta_i = sum(THETA[:, 1] * cfg.model_options.dz)
    theta_t = sum(THETA[:, -1] * cfg.model_options.dz)
    theta_tot = theta_i - theta_t  # (m)
    theta_tot = theta_tot * 1000  # (mm)

    infilt_tot = sum(infiltration) * dt * 1000  # mm
    if cfg.model_options.UpperBC == 0:
        theta_tot = (theta_tot) + infilt_tot
    ############################

    EVsink_total = np.zeros(shape=(len(EVsink_ts[0])))
    for i in np.arange(1, len(EVsink_ts[0]), 1):
        EVsink_total[i] = sum(-EVsink_ts[:, i] * dz)  # (1/s) over the simulation times dz [m]= m

    root_water = sum(EVsink_total) * 1000 * dt  # mm
    #############################

    # trans_2d [ m3H2O m-2ground s-1 m-1stem]
    transpiration_tot = sum(sum(trans_2d)) * 1000 * dt * dz  ##mm

    df_waterbal = pd.DataFrame(
        data={
            "theta_i": theta_i,
            "theta_t": theta_t,
            "theta_tot": theta_tot,
            "infilt_tot": infilt_tot,
            "root_water": root_water,
            "transpiration_tot": transpiration_tot,
        },
        index=[0],
    )

    ####################### Format model outputs ###################################
    # summing during all time steps and multiplying by 1000 = mm  #
    # the dt factor is accounting for the time step - to the TOTAl and not the rate

    # end of simulation adding +1 time step to match dimensions
    step_time = pd.Series(
        pd.date_range(start_time, end_time + pd.to_timedelta(dt, unit="s"), freq=str(dt) + "s")
    )
    ############################################################################

    # df_time = pd.DataFrame(data=step_time.index.values,index=step_time)

    #########################################################

    d = {"trans": (sum(trans_2d[:, :] * dz) * 1000)}  # mm/s
    df_EP = pd.DataFrame(data=d, index=step_time[:])

    ds_EP = xr.Dataset(
        data_vars=dict(TVeg=(["time"], df_EP.trans.values), Infiltration=(["time"], infiltration)),
        coords=dict(t=(["time"], df_EP.index)),
        attrs=dict(description="Model output"),
    )

    # datasets for output vars
    ds_EP = xr.Dataset(
        data_vars=dict(TVeg=(["time"], df_EP.trans.values), Infiltration=(["time"], infiltration)),
        coords=dict(t=(["time"], df_EP.index)),
        attrs=dict(description="Model output"),
    )

    # whole system
    ds_all = xr.Dataset(
        {
            "H": (["time", "z"], H.transpose(), dict(description="Water potential", units="MPa")),
            "K": (
                ["time", "z"],
                K.transpose(),
                dict(description="Hydraulic conductivity", units="m s-1"),
            ),
            "Capac": (
                ["time", "z"],
                Capac.transpose(),
                dict(description="Capacitance", units="Pa-1"),
            ),  # TODO
        },
        coords={
            "time": df_EP.index,
            "z": zind.z,
            "species": species
        },
    )

    # slice H, K, and Capac for soil, roots, and xylem
    ds_soil1, ds_root1, ds_canopy1 = slice_dsall(ds_all, zind)

    # soil
    ds_soil2 = xr.Dataset(
        {
            "THETA": (
                ["time", "z"],
                THETA.transpose(),
                dict(description="Volumetric water content", units="m3 m-3"),
            ),
        },
        coords={
            "time": df_EP.index,
            "z": zind.z_soil,
            "species": species
        },
    )
    # root TODO
    ds_root2 = xr.Dataset(
        {
            "Kr_sink": (
                ["time", "z"],
                Kr_sink.transpose(),
                dict(description="Effective root radial conductivity", units="1/sPa"),
            ),  # TODO
            "S_kr": (
                ["time", "z"],
                S_kr.transpose(),
                dict(
                    description="Cavitation of root xylem, given by eqn S.77 in Silva et al 2022",
                    units="unitless",
                ),
            ),
            "S_sink": (
                ["time", "z"],
                S_sink.transpose(),
                dict(
                    description=(
                        "Feddes root water uptake stress function, given by equations S.73, 74 and"
                        " 75 in Silva et al 2022"
                    ),
                    units="unitless",
                ),
            ),
            "EVsink_ts": (
                ["time", "z"],
                EVsink_ts.transpose(),
                dict(description="Root water uptake", units="m3H2O m-2ground m-1depth s-1"),
            ),
        },
        coords={
            "time": df_EP.index,
            "z": zind.z_root,
            "species": species
        },
    )
    # canopy
    ds_canopy2 = xr.Dataset(
        {
            "S_kx": (
                ["time", "z"],
                S_kx.transpose(),
                dict(
                    description="Cavitation of stem xylem, given by eqn S.79 in Silva et al 2022",
                    units="unitless",
                ),
            ),
            "trans_2d": (
                ["time", "z"],
                trans_2d.transpose(),
                dict(description="transpiration", units="m3H2O m-2crown_projection m-1stem s-1"),
            ),
            "nhl_trans_2d": (
                ["time", "z"],
                nhl_trans_2d.transpose(),
                dict(description="nhl transpiration", units="m3H2O m-2crown_projection m-1stem s-1"),
            ),
        },
        coords={
            "time": df_EP.index,
            "z": zind.z_upper,
            "species": species
        },
    )

    # Add H, K, and Capac to ds_soil, ds_roots, and ds_canopy
    ds_soil = xr.merge([ds_soil1, ds_soil2])
    ds_root = xr.merge([ds_root1, ds_root2])
    ds_canopy = xr.merge([ds_canopy1, ds_canopy2])

    return (
        df_waterbal,
        df_EP,
        {
            "ds_EP": ds_EP,
            "ds_soil": ds_soil,
            "ds_root": ds_root,
            "ds_canopy": ds_canopy,
            "ds_all": ds_all,
        },
    )


def slice_dsall(ds_all, zind):

    zind_soil = np.arange(0, zind.nz_s)
    zind_root = np.arange(zind.nz_s, zind.nz_r)
    zind_canopy = np.arange(zind.nz_r, zind.nz)

    ds_soil = ds_all.isel(z=zind_soil)
    ds_root = ds_all.isel(z=zind_root)
    ds_canopy = ds_all.isel(z=zind_canopy)

    return ds_soil, ds_root, ds_canopy


####################### Save model outputs ###################################
def save_csv(dir, df_waterbal, df_EP):
    # Writes model outputs to csv files

    # make output directory if one doesn't exist
    (dir).mkdir(exist_ok=True)

    # for var in output_vars:
    #     pd.DataFrame(output_vars[var]).to_csv(working_dir / 'output' / (var + '.csv'), index = False, header=False)

    df_waterbal.to_csv(dir / ("df_waterbal" + ".csv"), index=False, header=True)
    df_EP.to_csv(dir / ("df_EP" + ".csv"), index=True, header=True)


def save_nc(dir, xr_datasets):
    # Writes model output to netcdf file

    # make output directory if one doesn't exist
    (dir).mkdir(exist_ok=True)

    # save dataset
    for ds in xr_datasets:
        xr_datasets[ds].to_netcdf(dir / (ds + ".nc"))


def combine_outputs(results):
    """
    Combines the outputs for all species

    Parameters
    ----------
    results :

    Returns
    -------

    """
    soil = [sp['ds_soil'] for sp in results]
    root = [sp['ds_root'] for sp in results]
    canopy = [sp['ds_canopy'] for sp in results]
    sapflux = [sp['sapflux'] for sp in results]
    # all_ = [sp['ds_all'] for sp in results]

    ds_soil = xr.concat(soil, dim='species')
    ds_root = xr.concat(root, dim='species')
    ds_canopy = xr.concat(canopy, dim='species')
    ds_sapflux = xr.concat(sapflux, dim='species')

    # Calculate plot level sapflux
    ds_sapflux_tot = ds_sapflux[['sapflux_plot', 'storage_plot', 'delta_S_plot']].sum(dim='species', keep_attrs=True)
    ds_sapflux_tot = ds_sapflux_tot.assign_coords(species='plot_tot').expand_dims('species')

    # Add plot level sapflux to the dataset
    ds_sapflux = xr.merge([ds_sapflux, ds_sapflux_tot])

    return {
            "ds_soil": ds_soil,
            "ds_root": ds_root,
            "ds_canopy": ds_canopy,
            "ds_sapflux": ds_sapflux,
        }
