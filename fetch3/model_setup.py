
"""
###########
Model setup
###########

Spatial discretization to set up the model
"""
import numpy as np

from fetch3.model_config import cfg

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

dz = cfg.dz
z_soil, nz_s, z_root, nz_r, z_Above, nz_Above, z_upper, z, nz, nz_sand, nz_clay = spatial_discretization(
    cfg.dz, cfg.Soil_depth, cfg.Root_depth, cfg.Hspec, cfg.sand_d, cfg.clay_d)