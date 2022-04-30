from initial_conditions import initial_conditions
from model_functions import Picard

Picard = profile(Picard)
Picard(*initial_conditions())
