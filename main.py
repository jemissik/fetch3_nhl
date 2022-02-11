import numpy as np
import pandas as pd
from pathlib import Path

from model_config import cfg
from nhl_transpiration.NHL_functions import *

# Read in LAD and met data
# Met data must include
met_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / cfg.input_fname, parse_dates=[0])
LAD_data = pd.read_csv(Path.cwd() / 'nhl_transpiration/data' / cfg.LAD_norm)

met_data = met_data[(met_data.Timestamp >= pd.to_datetime(cfg.start_time)) &
                    (met_data.Timestamp <= pd.to_datetime(cfg.end_time))].reset_index(drop=True)

total_crown_area_sp = cfg.LAI * cfg.crown_scaling / cfg.sum_LAI_plot * cfg.plot_area

ds, LAD, zen = calc_NHL_timesteps(cfg.dz, cfg.Hspec, cfg.Cd, met_data, cfg.Vcmax25, cfg.alpha_gs, cfg.alpha_p,
            cfg.LAI, cfg.plot_area, total_crown_area_sp, cfg.mean_crown_area_sp, LAD_data[cfg.species], LAD_data.z_h,
            cfg.latitude, cfg.longitude, time_offset = cfg.time_offset)

write_outputs_netcdf(ds)
write_outputs({'zenith':zen, 'LAD': LAD})

#Interpolate to model time resolution
#time in seconds
ds2 = ds.assign_coords({'time': pd.to_timedelta(pd.to_datetime(ds.time.values) - pd.to_datetime(ds.time.values[0]))/ np.timedelta64(1,'s')})

# New time and space coordinates matching model resolution
model_ts = np.arange(0, len(ds.time) * cfg.dt + cfg.dt0, cfg.dt0)
model_z = np.arange(0, cfg.Hspec, cfg.dz)

#NHL transpiration in units of m s-1 * LAD  = kg H2O s-1 m-1stem m-2ground
da = ds2.NHL_trans_sp_stem #NHL in units of m s-1 * m-1stem
# da = ds2.NHL_trans_sp_stem * 10**-3

NHL_modelres = da.interp(z = model_z, time = model_ts, assume_sorted = True, kwargs={'fill_value':0})

#write NHL output to netcdf
NHL_modelres.to_netcdf('output/nhl_modelres_trans_out.nc')
NHL_modelres = NHL_modelres.data.transpose()