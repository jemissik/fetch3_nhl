# from FETCH2_loading_LAD import *
# from fetch3.met_data import *
# from fetch3.initial_conditions import *
# from jarvis import *
# from canopy import *

# from FETCH2_run_LAD import *
from fetch3.model_functions import *
from fetch3.initial_conditions import initial_conditions

import cProfile
import pstats

with cProfile.Profile() as pr:
    Picard(*initial_conditions())

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename = 'speed_test/output/cProfile_speed.prof')