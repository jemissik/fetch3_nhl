from FETCH2_config import *
from FETCH2_loading_LAD import *
from met_data import *

###################################################################
#STOMATA REDUCTIONS FUNCTIONS
#for transpiration formulation
#stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################

def jarvis_fs(SW_in):
    return 1-np.exp(-params['kr']*SW_in) #radiation
def jarvis_fTa(Ta):
    return 1-params['kt']*(Ta-params['Topt'])**2 #temperature
def jarvis_fd(VPD):
    return 1/(1+VPD*params['kd'])     #VPD

f_s=jarvis_fs(SW_in)
f_Ta=jarvis_fTa(Ta)
f_d=jarvis_fd(VPD)

#########################################################################3
#2D stomata reduction functions and variables for canopy-distributed transpiration
#############################################################################
#TODO convert to functions
f_Ta_2d=np.zeros(shape=(len(z_upper),len(f_Ta)))
for i in np.arange(0,len(f_Ta),1):
    f_Ta_2d[:,i]=f_Ta[i]
    if any(f_Ta_2d[:,i]<= 0):
        f_Ta_2d[:,i]=0


f_d_2d=np.zeros(shape=(len(z_upper),len(f_d)))
for i in np.arange(0,len(f_d),1):
    f_d_2d[:,i]=f_d[i]
    if any(f_d_2d[:,i]<= 0):
        f_d_2d[:,i]=0

f_s_2d=np.zeros(shape=(len(z_upper),len(f_d)))
for i in np.arange(0,len(f_s),1):
    f_s_2d[:,i]=f_s[i]
    if any(f_s_2d[:,i]<= 0):
        f_s_2d[:,i]=0


#2D INTERPOLATION NET RADIATION
NET_2d=np.zeros(shape=(len(z_upper),len(NET)))
for i in np.arange(0,len(NET),1):
    NET_2d[:,i]=NET[i]
    if any(NET_2d[:,i]<= 0):
        NET_2d[:,i]=0

#2D INTERPOLATION VPD
VPD_2d=np.zeros(shape=(len(z_upper),len(VPD)))
for i in np.arange(0,len(VPD),1):
    VPD_2d[:,i]=VPD[i]