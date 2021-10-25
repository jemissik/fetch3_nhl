from met_data import *

#######################################################################
#INITIAL CONDITIONS
#######################################################################
#soil initial conditions as described in the paper [VERMA et al., 2014]

def initial_conditions():
    initial_H=np.zeros(shape=nz)

    cte_clay = params['cte_clay']
    H_init_soilbottom = params['H_init_soilbottom']
    H_init_soilmid = params['H_init_soilmid']
    H_init_canopytop = params['H_init_canopytop']
    BottomBC = params['BottomBC']


    factor_soil=(H_init_soilbottom-(H_init_soilmid))/(int((params['clay_d']-cte_clay)/dz)) #factor for interpolation

    #soil
    for i in np.arange(0,len(z_soil),1):
        if  0.0<=z_soil[i]<=cte_clay :
            initial_H[i]=H_init_soilbottom
        if cte_clay<z_soil[i]<=z[nz_clay]:
            initial_H[i]=initial_H[i-1]-factor_soil #factor for interpolation
        if params['clay_d']<z_soil[i]<= z[nz_r-1]:
            initial_H[i]=H_init_soilmid

    initial_H[nz_s-1]=H_init_soilmid


    factor_xylem=(H_init_canopytop-(H_init_soilbottom))/((z[-1]-z[nz_s])/dz)

    #roots and xylem
    initial_H[nz_s]=H_init_soilbottom
    for i in np.arange(nz_s+1,nz,1):
        initial_H[i]=initial_H[i-1]+factor_xylem #meters


    #putting initial condition in Pascal
    H_initial=initial_H*params['g']*params['Rho']  #Pascals


    ###########################################################################
    #BOTTOM BOUNDARY CONDITION FOR THE SOIL
    #The model contains different options, therefore this variable is created but
    #only used if you choose a  Dirichlet BC
    ######################################################################
    soil_bottom=np.zeros(shape=len(q_rain))
    for i in np.arange(0,len(q_rain),1):
        soil_bottom[i]=28      #0.28 m3/m3 fixed moisture according to VERMA ET AL., 2014

    #clay - van genuchten
    Head_bottom=((((params['theta_R1']-params['theta_S1'])/(params['theta_R1']-(soil_bottom/100)))**(1/params['m_1'])-1)**(1/params['n_1']))/params['alpha_1']
    Head_bottom_H=-Head_bottom*params['g']*params['Rho']  #Pa
    Head_bottom_H=np.flipud(Head_bottom_H) #model starts the simulation at the BOTTOM of the soil

    ############## inital condition #######################
    #setting profile for initial condition
    if BottomBC==0:
        H_initial[0]=Head_bottom_H[0]

    return H_initial, Head_bottom_H