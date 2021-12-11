import numpy as np
import pandas as pd
from pathlib import Path

import nhl_transpiration.nhl_config as ncfg
# from NHL_functions import *
from nhl_transpiration.NHL_functions import *

import time
start = time.time()

# Example script to run model for the 2011 test data for ES

# Read in LAD and met data

met_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / ncfg.met_data, parse_dates=[0])
LAD_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / ncfg.LAD_norm)

# To run in NHL directory
# met_data = pd.read_csv(Path.cwd() / 'data' / ncfg.met_data, parse_dates=[0])
# LAD_data = pd.read_csv(Path.cwd() / 'data' / ncfg.LAD_norm)

met_data = met_data[(met_data.Timestamp >= pd.to_datetime(ncfg.start_time)) &
                    (met_data.Timestamp <= pd.to_datetime(ncfg.end_time))].reset_index(drop=True)
total_LAI_sp = np.array([1.1,1.45,0.84,0.044])*1.176*1.1 # vector, total leaf area index for each species [m2-leaf/m2-ground]

# TODO bs factor - will need to be altered. Overwriting the actual total crown area for each species
crown_scaling = np.array([2, 0.2, 0.1, 8])
total_crown_area_sp = total_LAI_sp * crown_scaling / sum(total_LAI_sp * crown_scaling) * ncfg.plot_area

#trim met data for a shorter run
# n = 20
# met_data = met_data.loc[7218:7218]


ds, tot_trans, LAD, zen = calc_NHL_timesteps(ncfg.dz, ncfg.height_sp, ncfg.Cd, met_data, ncfg.Vcmax25, ncfg.alpha_gs, ncfg.alpha_p,
            total_LAI_sp[0], ncfg.plot_area, total_crown_area_sp[0], ncfg.mean_crown_area_sp, LAD_data[ncfg.species], LAD_data.z_h,
            ncfg.latitude, ncfg.longitude, time_offset = ncfg.time_offset)


write_outputs_netcdf(ds)
write_outputs({'tot_trans':tot_trans, 'zenith':zen, 'LAD': LAD})

df = pd.DataFrame(data={'tot_trans':tot_trans, 'zenith':zen, 'Timestamp': met_data.Timestamp} )

#Interpolate to model time resolution
#time in seconds
ds2 = ds.assign_coords({'time': pd.to_timedelta(pd.to_datetime(ds.time.values) - pd.to_datetime(ds.time.values[0]))/ np.timedelta64(1,'s')})

# New time and space coordinates matching model resolution
# model_ts = pd.date_range(start = met_data.Timestamp[0], end = met_data.Timestamp.iloc[-1], freq = str(ncfg.dt0) + 's')
# model_ts = pd.date_range(start = ncfg.start_time, end = ncfg.end_time, freq = str(ncfg.dt0) + 's')
# model_ts = np.arange(ds2.time.values[0], ds2.time.values[-1] +  + ncfg.met_dt, ncfg.dt0)
model_ts = np.arange(0, len(ds.time) * ncfg.met_dt + ncfg.dt0, ncfg.dt0)

# model_z = np.arange(0, ncfg.height_sp + ncfg.dz, ncfg.dz)
model_z = np.arange(0, ncfg.height_sp, ncfg.dz)


#NHL transpiration in units of m s-1 * LAD  = kg H2O s-1 m-1stem m-2ground
da = ds2.NHL_trans_sp_stem * 10**-3 #NHL in units of m s-1 * m-1stem

NHL_modelres = da.interp(z = model_z, time = model_ts, assume_sorted = True, kwargs={'fill_value':0})
#write NHL output to netcdf
NHL_modelres.to_netcdf('output/nhl_modelres_trans_out.nc')

NHL_modelres = NHL_modelres.data.transpose()

print(f"run time: {time.time() - start} s")  # end run clock
