from pathlib import Path
import pandas as pd
import numpy as np

import model_config as cfg
from model_setup import z_upper, interpolate_2d

# Helper functions
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
def calc_infiltration_rate(precipitation, tmax, dt0):
    precipitation=precipitation/cfg.dt #dividing the value over half hour to seconds [mm/s]
    rain=precipitation/cfg.Rho  #[converting to m/s]
    q_rain=np.interp(np.arange(0,tmax+dt0,dt0), t_data, rain) #interpolating
    q_rain=np.nan_to_num(q_rain) #m/s precipitation rate= infiltration rate
    return q_rain
def calc_esat(Ta):
    return 611*np.exp((17.27*(Ta-273.15))/(Ta-35.85)) #Pascal
def calc_delta(Ta, e_sat):
    return (4098/((Ta-35.85)**2))*e_sat
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
df = pd.read_csv(data_path)
step_time_hh = pd.Series(pd.date_range(start_time, end_time, freq=str(cfg.dt)+'s'))

# Select data for length of run
df = df.iloc[0:len(step_time_hh)]
df.index = step_time_hh

tmax = len(df) * cfg.dt
t_data = np.arange(cfg.tmin, tmax, cfg.dt)         # data time grids for input data
t_data=list(t_data)
nt_data=len(t_data)                      #length of input data

#variables to arrays
precipitation = df['Rain (mm)'].values
Ta_C = df['T(degC)']
SW_in = df['Radiation (W/m2)']
VPD = df['VPD (kPa)']

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

########################################################################
#INTERPOLATING VARIABLES FOR PENMAN-MONTEITH TRANSPIRATION
#variables are in the data resolution (half-hourly) and are interpolated to model resolution
##########################################################################


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