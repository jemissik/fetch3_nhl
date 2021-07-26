
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:07:38 2019

@author: mdef0001
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


#This code is a simple example replicating the results of the topic
#3.3 Modeling LAD and capacitance from the paper:
#Tree Hydrodynamic Modelling of Soil Plant Atmosphere Continuum (SPAC-3Hpy)
#published at Geoscientific Model Development (gmd)
#contact marcela.defreitassilva@monash.edu and edoardo.daly@monash.edu
    
#Here we simulate the same conditions as in the study Verma et al., 2014, but
#considering a simplified LAD and capacitance formulation according to 
#Lalic, B. & Mihailovic, D. T. An Empirical Relation Describing 
#Leaf-Area Density inside the Forest for Environmental Modeling 
#Journal of Applied Meteorology, American Meteorological Society, 2004, 43, 641-645

#refer to the case study paper below for details on the model set up
#Verma, P.; Loheide, S. P.; Eamus, D. & Daly, E.
#Root water compensation sustains transpiration rates in an Australian woodland 
#Advances in Water Resources, Elsevier BV, 2014, 74, 91-101




############################## Opening files as dataframes #######################

BASE = Path.cwd()
path_1= BASE / "Precipitation_Verma.csv"
df_rain=pd.read_csv(path_1, float_precision='high',header=None)

path_2 = BASE / "Derek_data_up.csv"

date1 = pd.to_datetime('2007-01-01 00:00:00') #begining of simulation
date2=pd.to_datetime('2007-06-09 00:00:00')   #end of simulation 
step_time_hh = pd.Series(pd.date_range(date1, date2, freq='30T'))

df_verma = pd.read_csv(path_2)
df_verma.index = step_time_hh
df_rain.index = step_time_hh


Precipitation=df_rain.values
Precipitation=Precipitation.reshape(len(Precipitation))

params={}
#############################################################################
#MODEL PARAMETERS
#Values according to Verma et al., 2014

############################################################################


#OTHER PARAMETERS
params['Rho']=1000       #[kg/m3]
params['g']=9.8          #[m/s2]

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
theta_1_clay=0.08
theta_2_clay=0.12
            
theta_1_sand=0.05
theta_2_sand=0.09


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
#########################################################################3
#PENMAN-MONTEITH EQUATION PARAMETERS
###########################################################################
#W m^-2 is the same as J s^-1 m^-2
#1J= 1 kg m2/s2
#therefore 1W/m2 = kg/s3

kr=5*10**(-3)         #m2/W Jarvis radiation parameter
kt=1.6*10**(-3)       #K-2  Jarvis temperature parameter
Topt=289.15           #K   Jarvis temperature parameter (optimum temperature)
kd=1.1*10**(-3)       #Pa-1 Jarvis vapor pressure deficit temperature
hx50=-1274000         #Pa  Jarvis leaf water potential parameter



nl=2                   #[-] Jarvis leaf water potential parameter
gsmax=10*10**(-3)      #m/s Maximum leaf stomatal conductance
gb=2*10**(-2)          #m/s Leaf boundary layer conductance
LAI=1.5                #[-] Leaf area index
Cp=1200                # J/m3 K Heat capacity of air
ga=2*10**(-2)          #m/s Aerodynamic conductance
lamb=2.51*10**9        #J/m3 latent heat of vaporization 
gama=66.7              #Pa/K psuchrometric constant
Emax=1*10**(-9)        #m/s maximum nightime transpiration


#linear interpolation to make sure there are no gaps in the dataset

#temperature
Ta=df_verma['T(degC)']
Ta=(Ta+ 273.15) #converting temperature from degree Celsius to Kelvin
Ta=Ta.interpolate(method='linear')

#incoming solar radiation
S=df_verma['Radiation (W/m2)']
S=S.interpolate(method='time')

#Vapor pressure deficit
VPD=df_verma['VPD (kPa)'][(df_verma['VPD (kPa)']>0)] #eliminating negative VPD
VPD = VPD.reindex(S.index) 
VPD=VPD.interpolate(method='linear')*1000  #kPa to Pa
VPD=VPD.fillna(0)

###############################################################################
#NUMERICAL SOLUTION TIME AND SPACE CONSTANTS (dz and dt0)
###############################################################################
#The finite difference discretization constants
dt0=20 #model temporal resolution
dz=0.1 #model spatial resolution
stop_tol=0.0001  #stop tollerance of equation converging


###############################################################################
#TIME CONSTANTS (data resolution)
##########################Time constants#######################################
 
tmin =int(0)                             # tmin [s]
dt=1800 #seconds - input data resolution
tmax = (len(Ta)*dt)              
t_data = np.arange(tmin,tmax,dt)         # data time grids for input data
t_data=list(t_data)                        
nt_data=len(t_data)                      #length of input data   

###############################################################################

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

UpperBC=0
BottomBC=0


#SOIL SPATIAL DISCRETIZATION
Root_depth=3.2 #[m] depth of root collumn
Soil_depth=5   #[m]depth of soil collumn


##########################################
#below-ground spatial discretization
#######################################

zmin=0     #[m] minimum depth of soil [bottom of soil] 
z_soil=np.arange(zmin,Soil_depth+dz,dz )
nz_s=len(z_soil)

#measurements dephts of soil [m]
z_root=np.arange((Soil_depth-Root_depth)+zmin,Soil_depth+dz,dz )
nz_r=len(z_soil)+len(z_root)


#############################################
#above-ground spatial discretization
#################################################

z_Above=np.arange(zmin, params['Hspec']+dz, dz)  #[m]
nz_Above=len(z_Above)
z_upper=np.arange((z_soil[-1]+dz),(z_soil[-1]+params['Hspec']+dz),dz)

z=np.concatenate((z_soil,z_root,z_upper))

nz=len(z) #total number of nodes

####################################################################
#CONFIGURATION OF SOIL DUPLEX
#depths of layer/clay interface
#####################################################################
sand_d=5.0 #4.2----top soil #m
clay_d=4.2 #0------4.2 #m

nz_sand=int(np.flatnonzero(z==sand_d)) #node where sand layer finishes
nz_clay=int(np.flatnonzero(z==clay_d)) #node where clay layer finishes- sand starts


########################################################
#SETTING PRECIPITATION AS INFILTRATION BOUNDARY CONDITION
#in case of set by user
###########################################################

Precipitation=Precipitation/1800 #dividing the value over half hour to seconds [mm/s]
rain=Precipitation/params['Rho']  #[converting to m/s]
q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate

########################################################################
#INTERPOLATING VARIABLES FOR PENMAN-MONTEITH TRANSPIRATION
#variables are in the data resolution (half-hourly) and are interpolated to model resolution
##########################################################################

Ta=np.interp(np.arange(0,tmax+dt0,dt0), t_data, Ta)

S=np.interp(np.arange(0,tmax+dt0,dt0), t_data, S)

VPD=np.interp(np.arange(0,tmax+dt0,dt0), t_data, VPD)

e_sat=611*np.exp((17.27*(Ta-273.15))/(Ta-35.85)) #Pascal

delta=(4098/((Ta-35.85)**2))*e_sat
delta_2d=delta

##############################################################
#NET RADIATION
#In this case NET radiation is set as 60% of total incoming solar radtiation
#################################################################
NET=S*0.6

###################################################################
#STOMATA REDUCTIONS FUNCTIONS 
#for transpiration formulation
#stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################3
f_s=1-np.exp(-kr*S) #radiation

f_Ta=1-kt*(Ta-Topt)**2 #temperature

f_d=1/(1+VPD*kd)     #VPD

#########################################################################3 
#2D stomata reduction functions and variables for canopy-distributed transpiration
#############################################################################

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

#######################################################################
#LEAF ARE DENSITY FORMULATION (LAD) [1/m]
#######################################################################
#Simple LAD formulation to illustrate model capability
#following Lalic et al 2014
####################
z_LAD=z_Above[1:]    
LAD=np.zeros(shape=(int(params['Hspec']/dz)))  #[1/m]

params['L_m']=0.4  #maximum value of LAD a canopy layer
params['z_m']=11   #height in which L_m is found [m]


#LAD function according to Lalic et al 2014
for i in np.arange(0,len(z_LAD),1):
    if  0.1<=z_LAD[i]<params['z_m']:
        LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**6)*np.exp(6*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
    if  params['z_m']<=z_LAD[i]<params['Hspec']:
        LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**0.5)*np.exp(0.5*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
    if z_LAD[i]==params['Hspec']:
        LAD[i]=0    

#######################################################################
#INITIAL CONDITIONS
#######################################################################

#soil initial conditions as described in the paper [VERMA et al., 2014]
initial_H=np.zeros(shape=nz)

#the initial conditions were constant -6.09 m drom 0-3 metres (from soil bottom)
#from 3 meters, interpolation of -6.09 m to -0.402 m between 3-4.2 m
#from 4,2 m [sand layer] cte value of -0.402 m 
#the conditions are specific for this case study and therefore the hardcoding below
 
cte_clay=3 #depth from 0-3m initial condition of clay [and SWC] is constante

factor_soil=(-6.09-(-0.402))/(int((clay_d-cte_clay)/dz)) #factor for interpolation

#soil
for i in np.arange(0,len(z_soil),1):
    if  0.0<=z_soil[i]<=cte_clay :
        initial_H[i]=-6.09
    if cte_clay<z_soil[i]<=z[nz_clay]:
        initial_H[i]=initial_H[i-1]-factor_soil #factor for interpolation
    if clay_d<z_soil[i]<= z[nz_r-1]:
        initial_H[i]=-0.402               


initial_H[nz_s-1]=-0.402



factor_xylem=(-23.3-(-6.09))/((z[-1]-z[nz_s])/dz)

#roots and xylem
initial_H[nz_s]=-6.09
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


#################################################################################
#INDEXING OF DATA  - create data frames using step_time as an index
#in case you want to create a pandas dataframe to your variables
#############################################################################
date1 = pd.to_datetime('2007-01-01 00:00:00') #begining of simulation
date2=pd.to_datetime('2007-06-09  00:30:00')  #end of simulation adding +1 time step to math dimensions
step_time = pd.Series(pd.date_range(date1, date2, freq='30T'))










