"""
FETCH3 is a tree hydrodynamic model for water fluxes across the soil-plant-atmosphere continuum. FETCH3 simulates water
transport through the soil, roots, and xylem as flow through porous media. The model resolves water potentials along the
vertical dimension, and stomatal response is linked to xylem water potential.

See FETCH3's doc pages for instructions about how to use the model



Overview of package structure:

model_config
- Reads configuration yml file
- #TODO conversions of parameters to the required format of the model

model_setup
- sets up spatial discretization

met_data
- imports and prepares met data for model

initial_conditions
- calculates initial water potential conditions
"""


try:
    from fetch3.__version__ import version

    __version__ = version
except ImportError:
    # Package not installed
    ___version__ = "0.0.0"