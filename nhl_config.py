"""
Configuration for running NHL as a standalone module

Running the model from the command line:
----------------------------------------
Run the model by running ``main_standalone.py``

To specify an input config file or output directory in a location other than the
default, a different config file and output directory can be specified as command
line arguments, for example::
      python3 main_standalone.py --config_path /Users/username/fetch3/user_model_config.yml --output_path /Users/username/fetch3/output/

If the arguments ``--config_path`` and ``--output_path`` are omitted when running the
model from the command line, the defaults will be used.
"""

import argparse
import logging
from pathlib import Path

import yaml
from dataclasses import dataclass

# Default paths for config file, input data, and model output
parent_path = Path(__file__).parent
default_config_path = parent_path / 'nhl_config.yml'
default_data_path = parent_path / 'data'
default_output_path = parent_path / 'output'

# Taking command line arguments for path of config file, input data, and output directory
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', nargs='?', default=default_config_path)
    parser.add_argument('--data_path', nargs='?', default=default_data_path)
    parser.add_argument('--output_path', nargs='?', default= default_output_path)
    args = parser.parse_args()
    config_file = args.config_path
    data_dir = args.data_path
    output_dir = Path(args.output_path)
except SystemExit:  # sphinx passing in args instead, using default.
    #use default options if invalid command line arguments are given
    config_file = default_config_path
    data_path = default_data_path
    output_dir = default_output_path

# If using the default output directory, create directory if it doesn't exist
if output_dir == default_output_path:
  (output_dir).mkdir(exist_ok=True)

model_dir = Path(__file__).parent.resolve() # File path of model source code

# Set up logging
log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename=output_dir / "nhl.log",
                    filemode="w",
                    format=log_format,
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__file__)


# Log the directories being used
logger.info("Using config file: " + str(config_file) )
logger.info("Using output directory: " + str(output_dir) )


# Dataclass to hold the config parameters
@dataclass
class ConfigParams:
    """ Dataclass to hold parameters from .yml file
    """

    # File for input met data
    input_fname: str

    start_time: str #begining of simulation
    end_time: str #end

    dt:  int  #seconds - input data resolution

    ###############################################################################
    #SITE INFORMATION
    ###############################################################################
    latitude:  float
    longitude:  float
    time_offset:  float #Offset from UTC time, e.g EST = UTC -5 hrs

    zenith_method: str

    ###############################################################################
    #NUMERICAL SOLUTION TIME AND SPACE CONSTANTS (dz and dt0)
    ###############################################################################
    #The finite difference discretization constants
    dt0:  int  #model temporal resolution [s]
    dz:  float  #model spatial resolution [m]


    # TREE PARAMETERS
    species:  str
    LAD_norm:  str #LAD data

    #TREE PARAMETERS
    Hspec: float                      #Height average of trees [m]
    LAI: float                       #[-] Leaf area index

    #########################################################################3
    #NHL PARAMETERS
    ###########################################################################

    crown_scaling:  float

    mean_crown_area_sp:  float
    total_crown_area_sp:  float
    plot_area:  float
    sum_LAI_plot:  float

    Cd:  float # Drag coefficient
    alpha_ml:  float  # Mixing length constant
    Cf:  float  #Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
    x:  float  #Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

    Vcmax25:  float
    alpha_gs:  float
    alpha_p:  float


#TODO convert to function
# Read configs from yaml file
logger.info("Reading config file" )

with open(config_file, "r") as yml_config:
    config_dict = yaml.safe_load(yml_config)

# Convert config dict to config dataclass
cfg = ConfigParams(**config_dict['model_options'], **config_dict['parameters'])