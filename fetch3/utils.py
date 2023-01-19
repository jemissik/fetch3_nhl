import numpy as np
import datetime as dt
from pathlib import Path

#############################################
# Helper functions
#################################################

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
    for i in np.arange(0, len(x), 1):
        x_2d[:, i] = x[i]
    return x_2d


def neg2zero(x):
    return np.where(x < 0, 0, x)


def get_dt_now_as_str(fmt: str = "%Y%m%dT%H%M%S") -> str:
    """get the datetime as now as a str.

    fmt : str
        Default format is file friendly.
        See `strftime documentation <https://docs.python.org/3/library/datetime.html
        #strftime-and-strptime-behavior>`_ for more information on choices.
    """
    return dt.datetime.now().strftime(fmt)

def make_experiment_directory(
    output_dir, experiment_name: str = "", append_timestamp: bool = True, exist_ok: bool = False
):
    ts = get_dt_now_as_str() if append_timestamp else ""
    exp_name = "_".join(name for name in [experiment_name, ts] if name)
    ex_dir = Path(output_dir).expanduser() / exp_name
    ex_dir.mkdir(exist_ok=exist_ok)
    return ex_dir
