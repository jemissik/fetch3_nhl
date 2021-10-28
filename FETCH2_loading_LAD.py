
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:07:38 2019

@author: mdef0001
"""
import numpy as np

from model_config import dz, Soil_depth, Root_depth, Hspec, sand_d, clay_d

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

def spatial_discretization(dz, Soil_depth, Root_depth, Hspec, sand_d, clay_d):
    ##########################################
    #below-ground spatial discretization
    #######################################
    zmin=0     #[m] minimum depth of soil [bottom of soil]
    z_soil=np.arange(zmin,Soil_depth + dz, dz)
    nz_s=len(z_soil)

    #measurements depths of soil [m]
    z_root=np.arange((Soil_depth - Root_depth) + zmin, Soil_depth + dz, dz)
    nz_r=len(z_soil)+len(z_root)

    #############################################
    #above-ground spatial discretization
    #################################################
    z_Above=np.arange(zmin, Hspec + dz, dz)  #[m]
    nz_Above=len(z_Above)
    z_upper=np.arange((z_soil[-1] + dz),(z_soil[-1] + Hspec + dz), dz)

    z=np.concatenate((z_soil,z_root,z_upper))

    nz=len(z) #total number of nodes

    ####################################################################
    #CONFIGURATION OF SOIL DUPLEX
    #depths of layer/clay interface
    #####################################################################
    nz_sand=int(np.flatnonzero(z==sand_d)) #node where sand layer finishes
    nz_clay=int(np.flatnonzero(z==clay_d)) #node where clay layer finishes- sand starts
    return z_soil, nz_s, z_root, nz_r, z_Above, nz_Above, z_upper, z, nz, nz_sand, nz_clay


#############################################
# Helper functions 
#################################################
# Function to do to 2d interpolation
def interpolate_2d(x, zdim):
    """
    Interpolates input to 2d for canopy-distributed values

    Parameters
    ----------
    x : [type]
        input
    zdim : [type]
        length of z dimension
    """
    x_2d = np.zeros(shape=(zdim, len(x)))
    for i in np.arange(0,len(x),1):
        x_2d[:,i]=x[i]
    return x_2d

def neg2zero(x):
    return np.where(x < 0, 0, x)

z_soil, nz_s, z_root, nz_r, z_Above, nz_Above, z_upper, z, nz, nz_sand, nz_clay = spatial_discretization(dz, Soil_depth, Root_depth, Hspec, sand_d, clay_d)