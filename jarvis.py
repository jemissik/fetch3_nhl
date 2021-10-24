from FETCH2_config import *
from FETCH2_loading_LAD import *
from met_data import *
from canopy import *

###################################################################
#STOMATA REDUCTIONS FUNCTIONS
#for transpiration formulation
#stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################

# Jarvis parameters
gsmax = params['gsmax']
kr = params['kr']
kt = params['kt']
Topt = params['Topt']
kd = params['kd']
hx50 = params['hx50']
nl = params['nl']

#Penman-Monteith parameters
gb = params['gb']
Cp = params['Cp']
ga = params['ga']
lamb = params['lamb']
gama = params['gama']

#nighttime transpiration parameter
Emax = params['Emax']

def jarvis_fs(SW_in):
    fs = 1-np.exp(-kr*SW_in) #radiation
    fs = neg2zero(fs)
    return fs
def jarvis_fTa(Ta):
    fTa = 1-kt*(Ta-Topt)**2 #temperature
    fTa = neg2zero(fTa)
    return fTa #temperature
def jarvis_fd(VPD):
    fd = 1/(1+VPD*kd)     #VPD
    fd = neg2zero(fd)
    return fd
def jarvis_fleaf(hn):
    return (1 + (hn / hx50) ** nl) ** (-1)

def calc_gs(f_Ta, f_d, f_s, f_leaf):
    return gsmax * f_d * f_Ta * f_s * f_leaf

def calc_gc(gs, gb):
    return (gs * gb) / (gs + gb)

def pm_trans(NET, delta, Cp, VPD, lamb, gc, ga):
    return ((NET * delta + Cp * VPD * ga) / (lamb * (delta * gc + gama * (ga + gc)))) * gc #[m/s]

def night_trans(f_Ta, f_d, f_leaf):
    # Eqn S.64
    return Emax * f_Ta * f_d * f_leaf #[m/s]

def calc_transpiration(SW_in, NET, delta, Cp, VPD, lamb, gb, ga, f_Ta, f_s, f_d, f_leaf):

    if SW_in > 5: #income radiation > 5 = daylight
        gs = calc_gs(f_Ta, f_d, f_s, f_leaf)
        gc = calc_gc(gs, gb)
        transpiration = pm_trans(NET, delta, Cp, VPD, lamb, gc, ga)
    else: #nighttime transpiration
        transpiration = night_trans(f_Ta, f_d, f_leaf)
    return transpiration * LAD #m/s * 1/m = [1/s]

#########################################################################3
#2D stomata reduction functions and variables for canopy-distributed transpiration
#############################################################################
f_Ta_2d = jarvis_fTa(Ta_2d)
f_d_2d = jarvis_fd(VPD_2d)
f_s_2d = jarvis_fs(SW_in_2d)