from fetch3.initial_conditions import initial_conditions
from fetch3.model_functions import Picard

Picard = profile(Picard)
Picard(*initial_conditions())
