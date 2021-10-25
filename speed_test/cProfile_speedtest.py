# from FETCH2_loading_LAD import *
# from met_data import *
# from initial_conditions import *
# from jarvis import *
# from canopy import *

# from FETCH2_run_LAD import *
from model_functions import *

import cProfile
import pstats

with cProfile.Profile() as pr:
    Picard(H_initial)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename = 'speed_test/output/cProfile_speed.prof')