"""
################
PM transpiration
################

Functions for the Penman-Monteith transpiration scheme
Uses Jarvis stomata reduction functions
"""
import numpy as np

from fetch3.utils import neg2zero

###################################################################
# STOMATA REDUCTIONS FUNCTIONS
# for transpiration formulation
# stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################


def jarvis_fs(SW_in, kr):
    fs = 1 - np.exp(-kr * SW_in)  # radiation
    fs = neg2zero(fs)
    return fs


def jarvis_fTa(Ta, kt, Topt):
    fTa = 1 - kt * (Ta - Topt) ** 2  # temperature
    fTa = neg2zero(fTa)
    return fTa  # temperature


def jarvis_fd(VPD, kd):
    fd = 1 / (1 + VPD * kd)  # VPD
    fd = neg2zero(fd)
    return fd


def jarvis_fleaf(hn, hx50, nl):
    return (1 + (hn / hx50) ** nl) ** (-1)


def calc_gs(gsmax, f_Ta, f_d, f_s, f_leaf):
    return gsmax * f_d * f_Ta * f_s * f_leaf


def calc_gc(gs, gb):
    return (gs * gb) / (gs + gb)


def pm_trans(NET, delta, Cp, VPD, lamb, gama, gc, ga):
    return ((NET * delta + Cp * VPD * ga) / (lamb * (delta * gc + gama * (ga + gc)))) * gc  # [m/s]


def night_trans(Emax, f_Ta, f_d, f_leaf):
    # Eqn S.64
    return Emax * f_Ta * f_d * f_leaf  # [m/s]


def calc_pm_transpiration(
    SW_in, NET, delta, Cp, VPD, lamb, gama, gb, ga, gsmax, Emax, f_Ta, f_s, f_d, f_leaf, LAD
):

    gs = calc_gs(gsmax, f_Ta, f_d, f_s, f_leaf)
    gc = calc_gc(gs, gb)

    conditions = [SW_in > 5, SW_in <= 5]
    outputs = [
        pm_trans(NET, delta, Cp, VPD, lamb, gama, gc, ga),
        night_trans(Emax, f_Ta, f_d, f_leaf),
    ]

    transpiration = np.select(conditions, outputs)

    return transpiration * LAD  # m/s * 1/m = [1/s]
