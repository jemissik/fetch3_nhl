"""
###################
Met data
###################
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fetch3.model_config import ConfigParams, TranspirationScheme
from fetch3.utils import interpolate_2d

def import_ameriflux_data(filein):
    """
    Imports AmeriFlux data (in ONEFLUX format) to a DataFrame.
    Parameters
    ----------
    filein : str
        Filepath to data

    Returns
    -------
    df: pd.DataFrame
        Flux data
    """

    df = pd.read_csv(filein, parse_dates=["TIMESTAMP_START", "TIMESTAMP_END"], na_values=["-9999"])

    return df


def prepare_ameriflux_data(filein, cfg: ConfigParams):

    df = import_ameriflux_data(filein)
    df = df.rename(columns={"TIMESTAMP_START": "Timestamp"})

    # Rename
    if cfg.model_options.met_column_labels is not None:
        df = df.rename(columns=cfg.model_options.met_column_labels)

    # Add VPD in kPa. VPD_F is in hPa
    df["VPD_F_kPa"] = df.VPD_F / 10

    if cfg.transpiration_scheme == TranspirationScheme.PM:
        varlist = ["Timestamp", "TA_F", "VPD_F_kPa", "P_F", "SW_IN_F"]
    elif cfg.transpiration_scheme == TranspirationScheme.NHL:
        varlist = [
            "Timestamp",
            "TA_F",
            "VPD_F_kPa",
            "P_F",
            "SW_IN_F",
            "WS_F",
            "USTAR",
            "PPFD_IN",
            "CO2_F",
            "PA_F",
        ]
    else:
        raise ValueError(f"Unknown transpiration scheme: {cfg.transpiration_scheme}")

    # Keep only variables needed
    df = df[varlist]

    # fill any gaps in the data
    df = df.set_index('Timestamp')
    df = df.interpolate(method='linear')
    # extrapolate in case there were missing values at the endpoints
    df = df.interpolate(fill_value="extrapolate", limit_direction="both")
    df = df.reset_index()

    # Select data for length of run
    df = df[(df.Timestamp >= cfg.model_options.start_time) & (df.Timestamp <= cfg.model_options.end_time)].reset_index(
        drop=True
    )

    return df


# Helper functions
def calc_model_time_grid(df, cfg: ConfigParams):
    tmax = len(df) * cfg.model_options.dt
    t_data = np.arange(cfg.model_options.tmin, tmax, cfg.model_options.dt)  # data time grids for input data
    t_data = list(t_data)
    nt_data = len(t_data)  # length of input data
    return tmax, t_data, nt_data


def calc_infiltration_rate(cfg: ConfigParams, precipitation, tmax, t_data):
    precipitation = precipitation / cfg.model_options.dt  # dividing the value over half hour to seconds [mm/s]
    rain = precipitation / cfg.Rho  # [converting to m/s]
    q_rain = np.interp(np.arange(0, tmax + cfg.model_options.dt0, cfg.model_options.dt0), t_data, rain)  # interpolating
    q_rain = np.nan_to_num(q_rain)  # m/s precipitation rate= infiltration rate

    # Reduce by frac_infiltration
    q_rain = cfg.parameters.frac_infiltration * q_rain
    return q_rain


def interp_to_model_res(var, tmax, t_data, dt0):
    return np.interp(np.arange(0, tmax + dt0, dt0), t_data, var)


def calc_esat(Ta):
    return 611 * np.exp((17.27 * (Ta - 273.15)) / (Ta - 35.85))  # Pascal


def calc_delta(Ta, e_sat):
    return (4098 / ((Ta - 35.85) ** 2)) * e_sat


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


def prepare_met_data(cfg: ConfigParams, data_dir, z_upper):
    ###########################################################
    # Load and format input data
    ###########################################################

    # Input file
    data_path = data_dir / cfg.model_options.input_fname

    start_time = pd.to_datetime(cfg.model_options.start_time)
    end_time = pd.to_datetime(cfg.model_options.end_time)

    # read input data
    df = prepare_ameriflux_data(data_path, cfg)

    df = df.set_index("Timestamp")

    tmax, t_data, nt_data = calc_model_time_grid(df, cfg)

    # variables to arrays
    precipitation = df["P_F"].values

    Ta_C = df["TA_F"]
    SW_in = df["SW_IN_F"]
    VPD = df["VPD_F_kPa"]

    # temperature
    Ta = Ta_C + 273.15  # converting temperature from degree Celsius to Kelvin
    Ta = Ta.interpolate(method="linear")

    # incoming solar radiation
    SW_in = SW_in.interpolate(method="time")

    # vapor pressure deficit
    VPD = VPD[VPD > 0]  # eliminating negative VPD
    VPD = VPD.reindex(SW_in.index)
    VPD = VPD.interpolate(method="linear") * 1000  # kPa to Pa
    VPD = VPD.fillna(0)

    ########################################################
    # SETTING PRECIPITATION AS INFILTRATION BOUNDARY CONDITION
    # in case of set by user
    ###########################################################

    q_rain = calc_infiltration_rate(cfg, precipitation, tmax, t_data)

    Ta = interp_to_model_res(Ta, tmax, t_data, cfg.model_options.dt0)
    SW_in = interp_to_model_res(SW_in, tmax, t_data, cfg.model_options.dt0)
    VPD = interp_to_model_res(VPD, tmax, t_data, cfg.model_options.dt0)

    e_sat = calc_esat(Ta)
    delta_2d = calc_delta(Ta, e_sat)

    NET = calc_NETRAD(SW_in)

    ####2d interpolation of met data
    NET_2d = interpolate_2d(NET, len(z_upper))
    VPD_2d = interpolate_2d(VPD, len(z_upper))
    Ta_2d = interpolate_2d(Ta, len(z_upper))
    SW_in_2d = interpolate_2d(SW_in, len(z_upper))

    met = Met(
        q_rain=q_rain,
        delta_2d=delta_2d,
        NET_2d=NET_2d,
        VPD_2d=VPD_2d,
        Ta_2d=Ta_2d,
        SW_in_2d=SW_in_2d,
    )

    return met, tmax, start_time, end_time


@dataclass
class Met:
    """
    Dataclass to hold met data
    """

    q_rain: np.ndarray
    delta_2d: np.ndarray
    NET_2d: np.ndarray
    VPD_2d: np.ndarray
    Ta_2d: np.ndarray
    SW_in_2d: np.ndarray
