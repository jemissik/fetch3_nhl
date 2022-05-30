# from FETCH2_loading_LAD import *
# from fetch3.met_data import *
# from fetch3.initial_conditions import *
# from jarvis import *
# from canopy import *

import cProfile
import pstats

from fetch3.initial_conditions import initial_conditions

# from FETCH2_run_LAD import *
from fetch3.model_functions import *

with cProfile.Profile() as pr:
    Picard(*initial_conditions())

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename = 'speed_test/output/cProfile_speed.prof')