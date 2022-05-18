
"""
###########
Model setup
###########

Spatial discretization to set up the model
"""
import numpy as np
from dataclasses import dataclass

###########################################################
#Discretization
###########################################################

def spatial_discretization(dz, Soil_depth, Root_depth, Hspec, sand_d, clay_d):

    ##########################################
    #below-ground spatial discretization
    #######################################
    zmin=0     #[m] minimum depth of soil [bottom of soil]
    z_soil=np.around(np.arange(zmin,Soil_depth + dz, dz), decimals=5) # Rounding to get rid of floating point precision error
    nz_s=len(z_soil)

    #measurements depths of soil [m]
    z_root= np.around(np.arange((Soil_depth - Root_depth) + zmin, Soil_depth + dz, dz), decimals=5)
    nz_r=len(z_soil)+len(z_root)

    #############################################
    #above-ground spatial discretization
    #################################################
    z_Above=np.around(np.arange(zmin, Hspec + dz, dz), decimals=5)  #[m]
    nz_Above=len(z_Above)

    z_upper=np.around(np.arange((z_soil[-1] + dz),(z_soil[-1] + Hspec + dz), dz), decimals=5)

    z=np.concatenate((z_soil,z_root,z_upper))

    nz=len(z) #total number of nodes

    ####################################################################
    #CONFIGURATION OF SOIL DUPLEX
    #depths of layer/clay interface
    #####################################################################
    nz_sand=int(np.flatnonzero(z_soil==sand_d)) #node where sand layer finishes
    nz_clay=int(np.flatnonzero(z_soil==clay_d)) #node where clay layer finishes- sand starts

    zind = Zind(z_soil=z_soil,
                nz_s=nz_s,
                z_root=z_root,
                nz_r=nz_r,
                z_Above=z_Above,
                nz_Above=nz_Above,
                z_upper=z_upper,
                z=z,
                nz=nz,
                nz_sand=nz_sand,
                nz_clay=nz_clay)

    return zind

def temporal_discretization(cfg, tmax):
    ##############Temporal discritization according to MODEL resolution
    t_num = np.arange(0,tmax+cfg.dt0,cfg.dt0)         #[s]
    nt = len(t_num)  #number of time steps
    return t_num, nt

@dataclass
class Zind:
    z_soil: np.ndarray
    nz_s: int
    z_root: np.ndarray
    nz_r: int
    z_Above: np.ndarray
    nz_Above: int
    z_upper: np.ndarray
    z: np.ndarray
    nz: int
    nz_sand: int
    nz_clay: int