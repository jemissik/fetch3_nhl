import cProfile
import pstats
from pathlib import Path

try:
    from fetch3.main import (
        Picard,
        initial_conditions,
        prepare_met_data,
        setup_config,
        spatial_discretization,
        temporal_discretization,
    )
    from fetch3.model_functions import *
except ImportError:
    import os

    fetch_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys

    sys.path.insert(0, fetch_dir)
    from fetch3.main import (
        Picard,
        initial_conditions,
        prepare_met_data,
        setup_config,
        spatial_discretization,
        temporal_discretization,
    )
    from fetch3.model_functions import *


Picard = profile(Picard)
vanGenuchten = profile(vanGenuchten)
Porous_media_xylem = profile(Porous_media_xylem)


(Path(__file__).parent / "output").resolve().mkdir(exist_ok=True, parents=True)


def profile_fetch():

    config_file = Path(__file__).resolve().parent.parent / "config_files" / "model_config.yml"
    data_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir = Path(__file__).resolve().parent.parent / "output"

    cfg = setup_config(config_file)

    ##########Set up spatial discretization
    zind = spatial_discretization(
        cfg.dz, cfg.Soil_depth, cfg.Root_depth, cfg.Hspec, cfg.sand_d, cfg.clay_d
    )
    ######prepare met data
    met, tmax, start_time, end_time = prepare_met_data(cfg, data_dir, zind.z_upper)

    t_num, nt = temporal_discretization(cfg, tmax)

    ############## Calculate initial conditions #######################
    H_initial, Head_bottom_H = initial_conditions(cfg, met.q_rain, zind)

    Picard(cfg, H_initial, Head_bottom_H, zind, met, t_num, nt, output_dir, data_dir)


# funcs =

# for func in fetch3.model_functions

profile_fetch = profile(profile_fetch)
profile_fetch()
