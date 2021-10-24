#PHYSICAL CONSTANTS
params = {}
params['Rho'] = 1000  ##[kg m-3]
params['g'] = 9.8 # [m s-2]
###############################################################################
#INPUT DATA FILE PARAMETERS
###############################################################################
params['input_fname'] = "Derek_data_test30Min.csv"
params['start_time'] = "2007-01-01 00:00:00" #begining of simulation
params['end_time'] = "2007-01-01 00:30:00" #end of simulation
params['dt'] = 1800  #seconds - input data resolution
params['tmin'] = 0  #tmin [s]

###############################################################################
#RUN OPTIONS
###############################################################################
params['print_run_progress'] = False #Print time steps calculated (printing slows down run)

###############################################################################
#NUMERICAL SOLUTION TIME AND SPACE CONSTANTS (dz and dt0)
###############################################################################
#The finite difference discretization constants
params['dt0'] = 20  #model temporal resolution [s]
params['dz'] = 0.1  #model spatial resolution [m]
params['stop_tol'] = 0.0001  #stop tolerance of equation converging

#############################################################################
#MODEL PARAMETERS
#Values according to Verma et al., 2014
############################################################################

#CONFIGURING SOIL BOUNDARY CONDITIONS
#Here the user can choose the desired contition by setting the numbers as
#described below

#The configuration used follows Verma et al. 2014

#############################################################################

#Upper Boundary condition

#1 = no flux (Neuman)
#0 = infiltration


#Bottom Boundary condition

#2 = free drainage
#1 = no flux (Neuman)
#0 = constant potential (Dirichlet)

params['UpperBC']=0
params['BottomBC']=0

#SOIL SPATIAL DISCRETIZATION
params['Root_depth']=3.2 #[m] depth of root column
params['Soil_depth']=5   #[m]depth of soil column

####################################################################
#CONFIGURATION OF SOIL DUPLEX
#depths of layer/clay interface
#####################################################################
params['sand_d']=5.0 #4.2----top soil #m
params['clay_d']=4.2 #0------4.2 #m

#soil initial conditions as described in the paper [VERMA et al., 2014]
#the initial conditions were constant -6.09 m drom 0-3 metres (from soil bottom)
#from 3 meters, interpolation of -6.09 m to -0.402 m between 3-4.2 m
#from 4,2 m [sand layer] cte value of -0.402 m
params['cte_clay']=3 #depth from 0-3m initial condition of clay [and SWC] is constant
params['H_init_soilbottom'] = -6.09
params['H_init_soilmid'] = -0.402
params['H_init_canopytop'] = -23.3


#SOIL PARAMETERS - USING VAN GENUCHTEN RELATIONSHIPS

#CLAY
params['alpha_1']=0.8                        #soil hydraulic parameter [1/m]
params['theta_S1']=0.55                      #saturated volumetric soil moisture content [-]
params['theta_R1']=0.068                     #residual volumetric soil moisture content [-]
params['n_1']=1.5                            #soil hydraulic parameter  [-]
params['m_1']=1-(1/params['n_1'])            #soil hydraulic parameter  [-]
params['Ksat_1']=1.94*10**(-7)               #saturated hydraulic conductivity  [m/s]

#SAND
params['alpha_2']=14.5
params['theta_S2']=0.47
params['theta_R2']=0.045
params['n_2']=2.4
params['m_2']=1-(1/params['n_2'])
params['Ksat_2']=3.45*10**(-5)

#Soil stress parameters
params['theta_1_clay']=0.08
params['theta_2_clay']=0.12

params['theta_1_sand']=0.05
params['theta_2_sand']=0.09

#ROOT PARAMETERS
#diving by Rho*g since Richards equation is being solved in terms of \Phi (Pa)
params['Kr']=(7.2*10**(-10))/(params['Rho']*params['g']) #soil-to-root radial conductance [m/sPa]
params['qz']=9                                           #unitless - parameter for the root mass distribution - Verma et al., 2014
params['Ksax']=(10**(-5))/(params['Rho']*params['g'])    #specific axial conductivity of roots  [ m/s]
params['Aind_r']=1                                       #m2 root xylem/m2 ground]

#XYLEM PARAMETERS
params['kmax']=(10**(-5))/(params['Rho']*params['g'])    #conductivity of xylem  [ m2/sPa]
params['ap']=2*10**(-6)                                  #xylem cavitation parameter [Pa-1]
params['bp']=-1.5*10**(6)                                #xylem cavitation parameter [Pa]
params['Aind_x']=8.62*10**(-4)                           #m2 xylem/m2 ground]
params['Phi_0']=5.74*10**8                               #From bohrer et al 2005
params['p']=20                                           #From bohrer et al 2005
params['sat_xylem']=0.573                                #From bohrer et al 2005

#TREE PARAMETERS
params['Hspec']=14                      #Height average of trees [m]
params['LAI']=1.5                       #[-] Leaf area index
params['Abasal']=8.62*10**(-4)          #[m2basal/m2-ground] xylem cross-sectional area and site surface ratio

#######################################################################
#LEAF AREA DENSITY FORMULATION (LAD) [1/m]
#######################################################################
params['L_m']=0.4  #maximum value of LAD a canopy layer
params['z_m']=11   #height in which L_m is found [m]
params['LAI']=1.5                #[-] Leaf area index

#########################################################################3
#PENMAN-MONTEITH EQUATION PARAMETERS
###########################################################################
#W m^-2 is the same as J s^-1 m^-2
#1J= 1 kg m2/s2
#therefore 1W/m2 = kg/s3

params['gb']=2*10**(-2)          #m/s Leaf boundary layer conductance
params['Cp']=1200                # J/m3 K Heat capacity of air
params['ga']=2*10**(-2)          #m/s Aerodynamic conductance
params['lamb']=2.51*10**9        #J/m3 latent heat of vaporization
params['gama']=66.7              #Pa/K psychrometric constant

#########################################################################3
#JARVIS PARAMETERS
###########################################################################

params['gsmax']=10*10**(-3)      #m/s Maximum leaf stomatal conductance
params['kr']=5*10**(-3)         #m2/W Jarvis radiation parameter
params['kt']=1.6*10**(-3)       #K-2  Jarvis temperature parameter
params['Topt']=289.15           #K   Jarvis temperature parameter (optimum temperature)
params['kd']=1.1*10**(-3)       #Pa-1 Jarvis vapor pressure deficit temperature
params['hx50']=-1274000         #Pa  Jarvis leaf water potential parameter
params['nl']=2                   #[-] Jarvis leaf water potential parameter
params['Emax']=1*10**(-9)        #m/s maximum nightime transpiration
