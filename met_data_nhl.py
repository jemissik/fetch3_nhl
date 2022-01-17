from pathlib import Path
import pandas as pd
import numpy as np

import model_config as cfg

# Helper functions

def calc_infiltration_rate(precipitation, tmax, dt0):
    precipitation=precipitation/cfg.dt #dividing the value over half hour to seconds [mm/s]
    rain=precipitation/cfg.Rho  #[converting to m/s]
    q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
    q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate
    return q_rain
def interp_to_model_res(var, tmax, dt0):
    return np.interp(np.arange(0, tmax + dt0, dt0), t_data, var)

###########################################################
#Load and format input data
###########################################################

#Input file
working_dir = Path.cwd()
data_path = working_dir / 'data' / cfg.input_fname

start_time = pd.to_datetime(cfg.start_time)
end_time = pd.to_datetime(cfg.end_time)

#read input data
df = pd.read_csv(data_path, parse_dates=[0])

# Select data for length of run
df = df[(df.Timestamp >=start_time) & (df.Timestamp <=end_time)]
df = df.set_index('Timestamp')

tmax = len(df) * cfg.dt
t_data = np.arange(cfg.tmin, tmax, cfg.dt)         # data time grids for input data
t_data=list(t_data)
nt_data=len(t_data)                      #length of input data

#variables to arrays
precipitation = df['P_F'].values

########################################################
#SETTING PRECIPITATION AS INFILTRATION BOUNDARY CONDITION
#in case of set by user
###########################################################

q_rain = calc_infiltration_rate(precipitation, tmax, cfg.dt0)