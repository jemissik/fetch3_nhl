import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

import nhl_config as ncfg
from NHL_functions import *

# Example script to run model for the 2011 test data for ES

# Read in LAD and met data

met_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / ncfg.met_data)
LAD_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / ncfg.LAD_norm)


total_LAI_sp = np.array([1.1,1.45,0.84,0.044])*1.176*1.1 # vector, total leaf area index for each species [m2-leaf/m2-ground]

# TODO bs factor - will need to be altered. Overwriting the actual total crown area for each species
crown_scaling = np.array([2, 0.2, 0.1, 8])
total_crown_area_sp = total_LAI_sp * crown_scaling / sum(total_LAI_sp * crown_scaling) * ncfg.plot_area

#trim met data for a shorter run
# n = 20
# met_data = met_data.loc[7218:7218]


ds, tot_trans, zen = calc_NHL_timesteps(ncfg.dz, ncfg.height_sp, ncfg.Cd, met_data, ncfg.Vcmax25, ncfg.alpha_gs, ncfg.alpha_p,
            total_LAI_sp[0], ncfg.plot_area, total_crown_area_sp[0], ncfg.mean_crown_area_sp, LAD_data[ncfg.species], LAD_data.z_h,
            ncfg.latitude, ncfg.longitude, time_offset = ncfg.time_offset)


write_outputs_netcdf(ds)
write_outputs({'tot_trans':tot_trans, 'zenith':zen})

df = pd.DataFrame(data={'tot_trans':tot_trans, 'zenith':zen, 'time': met_data.Time} )