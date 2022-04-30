"""
###################
Met data
###################
"""

from pathlib import Path
import pandas as pd
import numpy as np

from fetch3.model_config import cfg, model_dir, data_dir
from fetch3.model_setup import z_upper, interpolate_2d

# Helper functions

def calc_infiltration_rate(precipitation, tmax, dt0):
    precipitation=precipitation/cfg.dt #dividing the value over half hour to seconds [mm/s]
    rain=precipitation/cfg.Rho  #[converting to m/s]
    q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
    q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate
    return q_rain
def interp_to_model_res(var, tmax, dt0):
    return np.interp(np.arange(0, tmax + dt0, dt0), t_data, var)
def calc_esat(Ta):
    return 611*np.exp((17.27*(Ta-273.15))/(Ta-35.85)) #Pascal
def calc_delta(Ta, e_sat):
    return (4098/((Ta-35.85)**2))*e_sat
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
###########################################################
#Load and format input data
###########################################################

#Input file
data_path = data_dir / cfg.input_fname

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

Ta_C = df['TA_F']
SW_in = df['SW_IN_F']
VPD = df['VPD_kPa']

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

########################################################
#SETTING PRECIPITATION AS INFILTRATION BOUNDARY CONDITION
#in case of set by user
###########################################################

q_rain = calc_infiltration_rate(precipitation, tmax, cfg.dt0)

Ta = interp_to_model_res(Ta, tmax, cfg.dt0)
SW_in = interp_to_model_res(SW_in, tmax, cfg.dt0)
VPD = interp_to_model_res(VPD, tmax, cfg.dt0)

e_sat = calc_esat(Ta)
delta_2d = calc_delta(Ta, e_sat)

NET = calc_NETRAD(SW_in)

####2d interpolation of met data
NET_2d = interpolate_2d(NET, len(z_upper))
VPD_2d = interpolate_2d(VPD, len(z_upper))
Ta_2d = interpolate_2d(Ta, len(z_upper))
SW_in_2d = interpolate_2d(SW_in, len(z_upper))