
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:07:38 2019

@author: mdef0001
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from FETCH2_config import params

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

###########################################################
#Load and format input data
###########################################################

#Input file
working_dir = Path.cwd()
data_path = working_dir / params['input_fname']

#time constants - data resolution
tmin = params['tmin'] # tmin [s]
dt = params['dt'] #seconds - input data resolution

start_time = pd.to_datetime(params['start_time'])
end_time = pd.to_datetime(params['end_time'])

#read input data
df = pd.read_csv(data_path)
step_time_hh = pd.Series(pd.date_range(start_time, end_time, freq=str(dt)+'s'))
df.index = step_time_hh

tmax = len(df) * dt
t_data = np.arange(tmin, tmax, dt)         # data time grids for input data
t_data=list(t_data)
nt_data=len(t_data)                      #length of input data

#variables to arrays
precipitation = df['Rain (mm)'].values
Ta_C = df['T(degC)']
SW_in = df['Radiation (W/m2)']
VPD = df['VPD (kPa)']

#temperature
Ta = Ta_C + 273.15 #converting temperature from degree Celsius to Kelvin
Ta = Ta.interpolate(method = 'linear')

#incoming solar radiation
SW_in = SW_in.interpolate(method = 'time')

#vapor pressure deficit
VPD=VPD[VPD > 0] #eliminating negative VPD
VPD = VPD.reindex(SW_in.index)
VPD=VPD.interpolate(method='linear')*1000  #kPa to Pa
VPD=VPD.fillna(0)

###########################################################
#Discretization
###########################################################
dz = params['dz']
dt0 = params['dt0']

def spatial_discretization():
    ##########################################
    #below-ground spatial discretization
    #######################################
    zmin=0     #[m] minimum depth of soil [bottom of soil]
    z_soil=np.arange(zmin,params['Soil_depth']+dz,dz)
    nz_s=len(z_soil)

    #measurements depths of soil [m]
    z_root=np.arange((params['Soil_depth']-params['Root_depth'])+zmin,params['Soil_depth']+dz,dz)
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
    nz_sand=int(np.flatnonzero(z==params['sand_d'])) #node where sand layer finishes
    nz_clay=int(np.flatnonzero(z==params['clay_d'])) #node where clay layer finishes- sand starts
    return z_soil, nz_s, z_root, nz_r, z_Above, nz_Above, z_upper, z, nz, nz_sand, nz_clay

z_soil, nz_s, z_root, nz_r, z_Above, nz_Above, z_upper, z, nz, nz_sand, nz_clay = spatial_discretization()

########################################################
#SETTING PRECIPITATION AS INFILTRATION BOUNDARY CONDITION
#in case of set by user
###########################################################
def calc_infiltration_rate(precipitation):
    precipitation=precipitation/dt #dividing the value over half hour to seconds [mm/s]
    rain=precipitation/params['Rho']  #[converting to m/s]
    q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
    q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate
    return q_rain

q_rain = calc_infiltration_rate(precipitation)

########################################################################
#INTERPOLATING VARIABLES FOR PENMAN-MONTEITH TRANSPIRATION
#variables are in the data resolution (half-hourly) and are interpolated to model resolution
##########################################################################

def interp_to_model_res(var):
    return np.interp(np.arange(0, tmax + dt0, dt0), t_data, var)

Ta = interp_to_model_res(Ta)
SW_in = interp_to_model_res(SW_in)
VPD = interp_to_model_res(VPD)

def calc_esat(Ta):
    return 611*np.exp((17.27*(Ta-273.15))/(Ta-35.85)) #Pascal

e_sat = calc_esat(Ta)

def calc_delta(Ta, e_sat):
    return (4098/((Ta-35.85)**2))*e_sat

delta = calc_delta(Ta, e_sat)
delta_2d=delta

def calc_NETRAD(SW_in):
    """
    Calculate net radiation as 60% of total incoming solar radiation

    Parameters
    ----------
    SW_in : [W m-2]
        Total incoming solar radiation

    Returns
    -------
    [W m-2]
        Net radiation
    """
    return SW_in * 0.6

NET = calc_NETRAD(SW_in)

###################################################################
#STOMATA REDUCTIONS FUNCTIONS
#for transpiration formulation
#stomata conductance as a function of radiation, temp, VPD and Phi
#################################################################3
f_s=1-np.exp(-params['kr']*SW_in) #radiation

f_Ta=1-params['kt']*(Ta-params['Topt'])**2 #temperature

f_d=1/(1+VPD*params['kd'])     #VPD

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
#LEAF AREA DENSITY FORMULATION (LAD) [1/m]
#######################################################################
#Simple LAD formulation to illustrate model capability
#following Lalic et al 2014
####################
def calc_LAD(z_Above):

    z_LAD=z_Above[1:]
    LAD=np.zeros(shape=(int(params['Hspec']/dz)))  #[1/m]

    #LAD function according to Lalic et al 2014
    for i in np.arange(0,len(z_LAD),1):
        if  0.1<=z_LAD[i]<params['z_m']:
            LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**6)*np.exp(6*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
        if  params['z_m']<=z_LAD[i]<params['Hspec']:
            LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**0.5)*np.exp(0.5*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
        if z_LAD[i]==params['Hspec']:
            LAD[i]=0
        return LAD
LAD = calc_LAD(z_Above)

#######################################################################
#INITIAL CONDITIONS
#######################################################################
# TODO make function for soil initial conditions
#soil initial conditions as described in the paper [VERMA et al., 2014]
initial_H=np.zeros(shape=nz)

#the initial conditions were constant -6.09 m drom 0-3 metres (from soil bottom)
#from 3 meters, interpolation of -6.09 m to -0.402 m between 3-4.2 m
#from 4,2 m [sand layer] cte value of -0.402 m
#the conditions are specific for this case study and therefore the hardcoding below
# TODO change hardcoding so that it is more configurable

cte_clay=3 #depth from 0-3m initial condition of clay [and SWC] is constant

factor_soil=(-6.09-(-0.402))/(int((params['clay_d']-cte_clay)/dz)) #factor for interpolation

#soil
for i in np.arange(0,len(z_soil),1):
    if  0.0<=z_soil[i]<=cte_clay :
        initial_H[i]=-6.09
    if cte_clay<z_soil[i]<=z[nz_clay]:
        initial_H[i]=initial_H[i-1]-factor_soil #factor for interpolation
    if params['clay_d']<z_soil[i]<= z[nz_r-1]:
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
date1 = start_time #begining of simulation
date2 = end_time + pd.to_timedelta(dt, unit = 's')  #end of simulation adding +1 time step to math dimensions
step_time = pd.Series(pd.date_range(date1, date2, freq=str(dt)+'s'))