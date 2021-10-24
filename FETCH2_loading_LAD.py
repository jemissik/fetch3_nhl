
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
from met_data import *


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