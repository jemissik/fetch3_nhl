import model_config as cfg
from met_data import q_rain
from model_setup import nz, dz, z, nz_s, nz_r, nz_clay, z_soil

import numpy as np
#######################################################################
#INITIAL CONDITIONS
#######################################################################
#soil initial conditions as described in the paper [VERMA et al., 2014]

def initial_conditions():
    initial_H=np.zeros(shape=nz)

    factor_soil=(cfg.H_init_soilbottom-(cfg.H_init_soilmid))/(int((cfg.clay_d-cfg.cte_clay)/dz)) #factor for interpolation

    #soil
    for i in np.arange(0,len(z_soil),1):
        if  0.0<=z_soil[i]<=cfg.cte_clay :
            initial_H[i]=cfg.H_init_soilbottom
        if cfg.cte_clay<z_soil[i]<=z[nz_clay]:
            initial_H[i]=initial_H[i-1]-factor_soil #factor for interpolation
        if cfg.clay_d<z_soil[i]<= z[nz_r-1]:
            initial_H[i]=cfg.H_init_soilmid

    initial_H[nz_s-1]=cfg.H_init_soilmid


    factor_xylem=(cfg.H_init_canopytop-(cfg.H_init_soilbottom))/((z[-1]-z[nz_s])/dz)

    #roots and xylem
    initial_H[nz_s]=cfg.H_init_soilbottom
    for i in np.arange(nz_s+1,nz,1):
        initial_H[i]=initial_H[i-1]+factor_xylem #meters


    #putting initial condition in Pascal
    H_initial=initial_H*cfg.g*cfg.Rho  #Pascals


    ###########################################################################
    #BOTTOM BOUNDARY CONDITION FOR THE SOIL
    #The model contains different options, therefore this variable is created but
    #only used if you choose a  Dirichlet BC
    ######################################################################
    soil_bottom=np.zeros(shape=len(q_rain))
    for i in np.arange(0,len(q_rain),1):
        soil_bottom[i]=28      #0.28 m3/m3 fixed moisture according to VERMA ET AL., 2014

    #clay - van genuchten
    Head_bottom=((((cfg.theta_R1-cfg.theta_S1)/(cfg.theta_R1-(soil_bottom/100)))**(1/cfg.m_1)-1)**(1/cfg.n_1))/cfg.alpha_1
    Head_bottom_H=-Head_bottom*cfg.g*cfg.Rho  #Pa
    Head_bottom_H=np.flipud(Head_bottom_H) #model starts the simulation at the BOTTOM of the soil

    ############## inital condition #######################
    #setting profile for initial condition
    if cfg.BottomBC==0:
        H_initial[0]=Head_bottom_H[0]

    return H_initial, Head_bottom_H