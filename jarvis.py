from FETCH2_config import *
from FETCH2_loading_LAD import *
from met_data import *

###################################################################
#STOMATA REDUCTIONS FUNCTIONS
#for transpiration formulation
#stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################

gsmax = params['gsmax']
kr = params['kr']
kt = params['kt']
Topt = params['Topt']
kd = params['kd']
hx50 = params['hx50']
nl = params['nl']
Emax = params['Emax']

def jarvis_fs(SW_in):
    return 1-np.exp(-kr*SW_in) #radiation
def jarvis_fTa(Ta):
    return 1-kt*(Ta-Topt)**2 #temperature
def jarvis_fd(VPD):
    return 1/(1+VPD*kd)     #VPD

#########################################################################3
#2D stomata reduction functions and variables for canopy-distributed transpiration
#############################################################################
f_Ta_2d = interpolate_2d(jarvis_fTa(Ta), len(z_upper))
f_d_2d = interpolate_2d(jarvis_fd(VPD), len(z_upper))
f_s_2d = interpolate_2d(jarvis_fs(SW_in), len(z_upper))