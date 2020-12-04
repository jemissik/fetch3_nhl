
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:07:38 2019

@author: mdef0001
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import math


############################## Opening files as dataframes #######################

path_4= r'C:\Users\marce\Desktop\Verma_data\layers_soil\Kaxial_cte\Verma cavitation\EP_BC\codes_verma_dz\Precipitation_Verma.csv'
df_rain=pd.read_csv(path_4, float_precision='high',header=None)

path_5 = r'C:\Users\marce\Desktop\Verma_data\layers_soil\Kaxial_cte\Verma cavitation\EP_BC\codes_verma_dz\Derek_data_up.csv'

date1 = pd.to_datetime('2007-01-01 00:00:00') #begining of simulation
date2=pd.to_datetime('2007-06-09 00:00:00')   #end of simulation 
step_time_hh = pd.Series(pd.date_range(date1, date2, freq='30T'))

df_verma = pd.read_csv(path_5)
df_verma.index = step_time_hh
df_rain.index = step_time_hh

#To simulate a shorther period of time
#df_rain=df_rain.loc['2007-01-01 1:00:00':]
#df_verma=df_verma.loc['2007-01-01 1:00:00':]


Precipitation=df_rain.values
Precipitation=Precipitation.reshape(len(Precipitation))

params={}

#OTHER PARAMETERS
params['Rho']=1000       #[kg/m3]
params['g']=9.8          #[m/s2]

#SOIL PARAMETERS - USING VAN GENUCHTEN 

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


#ROOT PARAMETERS 
#diving by Rho*g since Richards equation is being solved in terms of \Phi (Pa)
params['Kr']=(7.2*10**(-10))/(params['Rho']*params['g']) #soil-to-root radial conductance [m/sPa]                                                          
params['qz']=9                                           #unitless - parameter for the root mass distribution - Verma et al., 2014
params['Ksax']=(10**(-5))/(params['Rho']*params['g'])    #specific axial conductivity of roots  [ m/s]


#XYLEM PARAMETERS
params['kmax']=(10**(-5))/(params['Rho']*params['g'])    #conductivity of xylem  [ m/s]
params['ap']=2*10**(-6)                                  #xylem cavitation parameter [Pa-1]
params['bp']=-1.5*10**(6)                                #xylem cavitation parameter [Pa]
params['Aind_x']=8.62*10**(-4)                           #m2 xylem/m2 ground]  
params['Phi_0']=5.74*10**8                               #From bohrer et al 2005       
params['p']=20                                           #From bohrer et al 2005
params['sat_xylem']=0.573                                #From bohrer et al 2005 
####################PENMAN-MONTEITH EQUATION PARAMETERS
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


#Interpolating input variables for PM transpiration

#temperature
Ta=df_verma['T(degC)']
Ta=(Ta+ 273.15)
Ta=Ta.interpolate(method='linear')

#incoming solar radiation
S=df_verma['Radiation (W/m2)']
S=S.interpolate(method='time')

#Vapor pressure deficit
VPD=df_verma['VPD (kPa)'][(df_verma['VPD (kPa)']>0)]
VPD = VPD.reindex(S.index) 
VPD=VPD.interpolate(method='linear')*1000  #kPa to Pa
VPD=VPD.fillna(0)

#######################The finite difference discretization constants######################
dt0=20 #model temporal resolution
dz=0.1 #model spatial resolution
stop_tol=0.0001
##########################Time constants#######################################
 
tmin =int(0)                             # tmin [s]
dt=1800 #s input data resolution
tmax = (len(Ta)*dt)              
t_data = np.arange(tmin,tmax,dt)         # data time grids for input data
t_data=list(t_data)                        
nt_data=len(t_data)                      #length of input data   

###############################################################################

#TREE PARAMETERS
params['Hspec']=14                      #Height average of trees [m]
params['LAI']=1.5                       #[-] Leaf area index                 
params['Abasal']=8.62*10**(-4)          #[m2basal/m2-ground] xylem cross-sectional area and site surface ratio


################### SOIL Boundary conditions

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


#############################below-ground spatial discretization

zmin=0     #[m] minimum depth of soil [bottom of soil] 
z_soil=np.arange(zmin,Soil_depth+dz,dz )
nz_s=len(z_soil)

#measurements dephts of soil [m]
z_root=np.arange((Soil_depth-Root_depth)+zmin,Soil_depth+dz,dz )
nz_r=len(z_soil)+len(z_root)


##############################above-ground spatial discretization
z_Above=np.arange(zmin, params['Hspec']+dz, dz)  #[m]
nz_Above=len(z_Above)
z_upper=np.arange((z_soil[-1]+dz),(z_soil[-1]+params['Hspec']+dz),dz)

z=np.concatenate((z_soil,z_root,z_upper))

nz=len(z) #total number of nodes

####################################################################
#CONFIGURATION OF SOIL DUPLEX
#depths of layer/clay interface
sand_d=5.0 #4.2----top soil #m
clay_d=4.2 #0------4.2 #m

nz_sand=int(np.flatnonzero(z==sand_d)) #node where sand layer finishes
nz_clay=int(np.flatnonzero(z==clay_d)) #node where clay layer finishes- sand starts


#####################Percipitation and temporal interpolation 

Precipitation=Precipitation/1800 #dividing the value over half hour to seconds [mm/s]
rain=Precipitation/params['Rho']  #[converting to m/s]
q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate


########################################################################
#VARIABLES FOR CALCULATING PM TRANSPIRATION

Ta=np.interp(np.arange(0,tmax+dt0,dt0), t_data, Ta)

S=np.interp(np.arange(0,tmax+dt0,dt0), t_data, S)

VPD=np.interp(np.arange(0,tmax+dt0,dt0), t_data, VPD)

e_sat=611*np.exp((17.27*(Ta-273.15))/(Ta-35.85)) #Pascal

delta=(4098/((Ta-35.85)**2))*e_sat
delta_2d=delta

################################################
NET=S*0.6

######################### reduction functions
f_s=1-np.exp(-kr*S) #radiation

f_Ta=1-kt*(Ta-Topt)**2 #temperature

f_d=1/(1+VPD*kd)     #VPD

########## 2D reduction functions and variables for canopy-distributed transpiration
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

####################Leaf area density formulation (LAD) [1/m]
z_LAD=z_Above[1:]    
LAD=np.zeros(shape=(int(params['Hspec']/dz)))  #[1/m]

params['L_m']=0.2  #maximum value of LAD a canopy layer
params['z_m']=11   #height in which L_m is found [m]

#LAD function according to Lalic et al 2014
#LAD function according to Lalic et al 2014
for i in np.arange(0,len(z_LAD),1):
    if  0.1<=z_LAD[i]<params['z_m']:
        LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**6)*np.exp(6*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
    if z_LAD[i]==params['z_m']:
        LAD[i]=0.2
    if  params['z_m']<z_LAD[i]<params['Hspec']:
        LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**0.5)*np.exp(0.5*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
    if z_LAD[i]==params['Hspec']:
        LAD[i]=0    



############ INITIAL conditions according to VERMA ET AL., 2014 
#soil initial conditions as described in the paper [VERMA et al., 2014]
#THIS WILL BE CHANGED FOR A MORE GENERAL APPROACH - DIFFERENT SITE 
initial_H=np.zeros(shape=nz)


factor_soil=5.688/(int((4.2-3)/dz)) #factor por interpolation

for i in np.arange(0,len(z_soil),1):
    if  0.0<=z_soil[i]<=3.0 :
        initial_H[i]=-6.09
    if 3.0<z_soil[i]<=4.2:
        initial_H[i]=initial_H[i-1]+factor_soil #factor for interpolation
    if 4.2<z_soil[i]<= 5:
        initial_H[i]=-0.402               


initial_H[nz_s-1]=-0.402

factor_xylem=17.21/((19-1.8)/dz)

#roots and xylem
initial_H[nz_s]=-6.09
for i in np.arange(nz_s+1,nz,1):
    initial_H[i]=initial_H[i-1]-factor_xylem #meters
    

H_initial=initial_H*params['g']*params['Rho']  #Pascals



##################### KNOWN MOISTURE soil bottom boundary condition
soil_bottom=np.zeros(shape=len(q_rain))
for i in np.arange(0,len(q_rain),1):  
    soil_bottom[i]=28      #0.28 m3/m3 fixed moisture according to VERMA ET AL., 2014

#clay - van genuchten
Head_bottom=((((params['theta_R1']-params['theta_S1'])/(params['theta_R1']-(soil_bottom/100)))**(1/params['m_1'])-1)**(1/params['n_1']))/params['alpha_1']
Head_bottom_H=-Head_bottom*params['g']*params['Rho']  #Pa
Head_bottom_H=np.flipud(Head_bottom_H) #model starts the simulation at the BOTTOM of the soil



#INDEXING OF DATA  - create data frames using step_time as an index

date1 = pd.to_datetime('2007-01-01 00:00:00') #begining of simulation
date2=pd.to_datetime('2007-06-09  00:30:00')  #end of simulation adding +1 time step of half hour to math dimensions
step_time = pd.Series(pd.date_range(date1, date2, freq='30T'))










