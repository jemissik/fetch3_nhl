import model_config as cfg
from model_setup import z_Above

import numpy as np
#######################################################################
#LEAF AREA DENSITY FORMULATION (LAD) [1/m]
#######################################################################
#Simple LAD formulation to illustrate model capability
#following Lalic et al 2014
####################
def calc_LAD(z_Above, dz, z_m, Hspec, L_m):

    z_LAD=z_Above[1:]
    LAD=np.zeros(shape=(int(Hspec/dz)))  #[1/m]

    #LAD function according to Lalic et al 2014
    for i in np.arange(0,len(z_LAD),1):
        if  0.1<=z_LAD[i]<z_m:
            LAD[i]=L_m*(((Hspec-z_m)/(Hspec-z_LAD[i]))**6)*np.exp(6*(1-((Hspec-z_m)/(Hspec-z_LAD[i]))))
        if  z_m<=z_LAD[i]<Hspec:
            LAD[i]=L_m*(((Hspec-z_m)/(Hspec-z_LAD[i]))**0.5)*np.exp(0.5*(1-((Hspec-z_m)/(Hspec-z_LAD[i]))))
        if z_LAD[i]==Hspec:
            LAD[i]=0
    return LAD
LAD = calc_LAD(z_Above, cfg.dz, cfg.z_m, cfg.Hspec, cfg.L_m)