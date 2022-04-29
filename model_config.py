"""
###################
Model configuration
###################

Reads in model configuration from .yml file

Model setup options and model parameters are read from a .yml file, which can be
modified by the user.


*************************
Modifying the config file
*************************

.yml file contents
==================
See ``model_config.yml`` for an example.

Model options
-------------
* **input_fname** (str): File for input met data
* **start_time** (str): ["YYYY-MM-DD HH:MM:SS"] Begining of simulation
* **end_time** (str): ["YYYY-MM-DD HH:MM:SS"] End of simulation

* **dt** (int): [seconds] Input data resolution
* **tmin** (int): [s]  #tmin #TODO

Site information
----------------
* **latitude** (float): Latitude of site in decimal degrees
* **longitude** (longitude): Longitude of site in decimal degrees
* **time_offset** (float): Offset from UTC time, e.g EST = UTC -5 hrs

Run options - printing
----------------------
Options to turn printing off or specify print frequency. Printing the run
progress will slow down the model run. The model will run faster if printing is
turned off or set to print more infrequently.

* **print_run_progress** (bool): Turns on/off printing for progress of the model run.
  ``print_run_progress: False`` will turn off printing.
* **print_freq** (int): Interval of timesteps to print if ``print_run_progress = True``
  (e.g. ``print_freq: 100`` will print every 100 time steps)

Transpiration options
---------------------

* **transpiration_scheme** (int): Whether to use the PM transpiration scheme or the NHL transpiration scheme

  * 0: PM transpiration
  * 1: NHL transpiration

Numerical solution time and space constants (dz and dt0)
--------------------------------------------------------
The finite difference discretization constants

* **dt0** (int): model temporal resolution [s]
* **dz** (float): model spatial resolution [m]
* **stop_tol** (float): stop tolerance of equation converging

Model parameters
----------------


Soil boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^
Here the user can choose the desired soil boundary conditions as described below.
The configuration used in the example config file follows Verma et al. 2014.

* **UpperBC** (int): Upper boundary condition
  Options:
    * 1: no flux (Neuman)
    * 0: infiltration
* **BottomBC** (int): Bottom boundary condition
  Options:
    * 2: free drainage
    * 1: no flux (Neuman)
    * 0: constant potential (Dirichlet)

Soil spatial discretization
^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Root_depth** (float): *[m]* depth of root column
* **Soil_depth** (float): *[m]* depth of soil column

Configuration of soil duplex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Depths of layer/clay interface

* **sand_d** (float): # TODO 4.2----top soil #m
* **clay_d** (float): Depth of clay layer. #0------4.2 #m #TODO

Soil initial conditions
^^^^^^^^^^^^^^^^^^^^^^^
Soil initial conditions as described in the paper [VERMA et al., 2014]
The initial conditions were constant -6.09 m drom 0-3 metres (from soil bottom)
from 3 meters, interpolation of -6.09 m to -0.402 m between 3-4.2 m
from 4,2 m [sand layer] cte value of -0.402 m

* **cte_clay** (float): depth from 0-3m initial condition of clay [and SWC] is constant
* **H_init_soilbottom** (float):  #TODO
* **H_init_soilmid** (float):  #TODO
* **H_init_canopytop** (float):  #TODO

Soil parameters - using Van Genuchten relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clay:

* **alpha_1** (float): soil hydraulic parameter *[1/m]*
* **theta_S1** (float): saturated volumetric soil moisture content *[-]*
* **theta_R1** (float): residual volumetric soil moisture content *[-]*
* **n_1** (float): soil hydraulic parameter *[-]*
* **m_1** (float): soil hydraulic parameter *[-]*. m_1 = 1-(1/n_1)
* **Ksat_1** (float): saturated hydraulic conductivity *[m/s]*

Sand: same definitions as above, but for sand

* **alpha_2** (float):
* **theta_S2** (float):
* **theta_R2** (float):
* **n_2** (float):
* **m_2** (float): m_2 = 1-(1/n_2)
* **Ksat_2** (float):

Soil stress parameters:

* **theta_1_clay** (float):
* **theta_2_clay** (float):
* **theta_1_sand** (float): float
* **theta_2_sand** (float): float

Tree parameters:
^^^^^^^^^^^^^^^^
* **species** (str):  Tree species


PHYSICAL CONSTANTS
^^^^^^^^^^^^^^^^^^
#TODO remove from config
**Rho**:  float  ##[kg m-3]
**g**:  float # [m s-2]

Root parameters
^^^^^^^^^^^^^^^
Dividing by Rho*g since Richards equation is being solved in terms of \Phi (Pa)

* **Kr** (float): soil-to-root radial conductance *[m/sPa]*, divided by rho*g
* **qz** (float): unitless - parameter for the root mass distribution - Verma et al., 2014
* **Ksax** (float): specific axial conductivity of roots *[m/s]*, divided by rho*g
* **Aind_r** (float): *[m2 root xylem/m2 ground]*

Xylem parameters
^^^^^^^^^^^^^^^^
* **kmax** (float): conductivity of xylem *[m2/sPa]*, divided by rho*g
* **ap** (float): xylem cavitation parameter *[Pa-1]*
* **bp** (float): xylem cavitation parameter *[Pa]*
* **Aind_x** (float): *[m2 xylem/m2 ground]*
* **Phi_0** (float): From bohrer et al 2005
* **p** (float): From bohrer et al 2005
* **sat_xylem** (float): From bohrer et al 2005

Tree parameters
^^^^^^^^^^^^^^^
* **Hspec** (float): Height average of trees *[m]*
* **LAI** (float): *[-]* Leaf area index

NHL transpiration scheme parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If using the NHL transpiration scheme.

* **crown_scaling** (float):

* **mean_crown_area_sp** (float):
* **total_crown_area_sp** (float):
* **plot_area** (float):
* **sum_LAI_plot** (float):

* **Cd** (float): Drag coefficient
* **alpha_ml** (float): Mixing length constant
* **Cf** (float): Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
* **x** (float): Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

* **Vcmax25** (float):
* **alpha_gs** (float):
* **alpha_p** (float):

* **wp_s50** (float): # TODO value for oak from Mirfenderesgi
* **c3** (float):  # TODO value for oak from Mirfenderesgi

* **LAD_norm** (str):  File with normalized LAD data

Penman-Monteith transpiration parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **lad_scheme** (int) : 0: default scheme, based on Lalic et al 2014; 1: scheme from NHL module

parameters if using penman-monteith transpiration scheme, based on Lalic et al 2014
if using NHL transpiration scheme, LAD is calculated in NHL module

* **L_m** (float): maximum value of LAD a canopy layer
* **z_m** (float): height in which L_m is found [m]


Penman-Monteith equation parameters
Note: W m^-2 is the same as J s^-1 m^-2
1J= 1 kg m2/s2
therefore 1W/m2 = kg/s3

* **gb** (float): *[m/s]* Leaf boundary layer conductance
* **Cp** (float): *[J/m3 K]* Heat capacity of air
* **ga** (float): *[m/s]* Aerodynamic conductance
* **lamb** (float): *[J/m3]* latent heat of vaporization
* **gama** (float): *[Pa/K]* psychrometric constant

Jarvis parameters

* **gsmax** (float): *[m/s]* Maximum leaf stomatal conductance
* **kr** (float): *[m2/W]* Jarvis radiation parameter
* **kt** (float): *[K-2]* Jarvis temperature parameter
* **Topt** (float): *[K]* Jarvis temperature parameter (optimum temperature)
* **kd** (float): *[Pa-1] Jarvis vapor pressure deficit temperature
* **hx50** (float): *[Pa]*  Jarvis leaf water potential parameter
* **nl** (float): *[-]* Jarvis leaf water potential parameter
* **Emax** (float): *[m/s]* maximum nightime transpiration

"""

import argparse
import logging
from pathlib import Path

import yaml
from dataclasses import dataclass

# Default paths for config file, input data, and model output
parent_path = Path(__file__).parent
default_config_path = parent_path / 'model_config.yml'
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
    data_dir = Path(args.data_path)
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

logging.basicConfig(filename=output_dir / "fetch3.log",
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
    ###############################################################################
    #INPUT DATA FILE PARAMETERS
    ###############################################################################

    # File for input met data
    input_fname: str

    start_time: str #begining of simulation
    end_time: str #end

    dt:  int  #seconds - input data resolution
    tmin:  int  #tmin [s]

    ###############################################################################
    #SITE INFORMATION
    ###############################################################################
    latitude:  float
    longitude:  float
    time_offset:  float #Offset from UTC time, e.g EST = UTC -5 hrs

    ###############################################################################
    #RUN OPTIONS - printing
    ###############################################################################
    # Printing slows down model run
    # Options to turn printing off or specify print frequency
    print_run_progress:  bool  # Turn on/off printing for progress of time steps calculated
    print_freq:  int  # Interval of timesteps to print if print_run_progress = True (e.g. 1 will print every time step)

    ###############################################################################
    #TRANSPIRATION OPTIONS - NHL OR PM
    ###############################################################################
    transpiration_scheme:  int # 0: PM transpiration; 1: NHL transpiration
    zenith_method:  str
    ###############################################################################
    #NUMERICAL SOLUTION TIME AND SPACE CONSTANTS (dz and dt0)
    ###############################################################################
    #The finite difference discretization constants
    dt0:  int  #model temporal resolution [s]
    dz:  float  #model spatial resolution [m]

    stop_tol:  float  #stop tolerance of equation converging

    #############################################################################
    #MODEL PARAMETERS
    #Values according to Verma et al., 2014
    ############################################################################

    #CONFIGURING SOIL BOUNDARY CONDITIONS
    #Here the user can choose the desired contition by setting the numbers as
    #described below

    #The configuration used follows Verma et al. 2014

    #############################################################################

    #Upper Boundary condition

    #1 = no flux (Neuman)
    #0 = infiltration


    #Bottom Boundary condition

    #2 = free drainage
    #1 = no flux (Neuman)
    #0 = constant potential (Dirichlet)

    UpperBC: int
    BottomBC: int

    #SOIL SPATIAL DISCRETIZATION

    Root_depth: float #[m] depth of root column
    Soil_depth: float   #[m]depth of soil column

    ####################################################################
    #CONFIGURATION OF SOIL DUPLEX
    #depths of layer/clay interface
    #####################################################################

    sand_d: float #4.2----top soil #m
    clay_d: float #0------4.2 #m

    #SOIL INITIAL CONDITIONS
    #soil initial conditions as described in the paper [VERMA et al., 2014]
    #the initial conditions were constant -6.09 m drom 0-3 metres (from soil bottom)
    #from 3 meters, interpolation of -6.09 m to -0.402 m between 3-4.2 m
    #from 4,2 m [sand layer] cte value of -0.402 m

    cte_clay: float #depth from 0-3m initial condition of clay [and SWC] is constant
    H_init_soilbottom:  float
    H_init_soilmid:  float
    H_init_canopytop:  float

    #SOIL PARAMETERS - USING VAN GENUCHTEN RELATIONSHIPS

    #CLAY
    alpha_1: float                        #soil hydraulic parameter [1/m]
    theta_S1: float                      #saturated volumetric soil moisture content [-]
    theta_R1: float                     #residual volumetric soil moisture content [-]
    n_1: float                            #soil hydraulic parameter  [-]
    #m_1 = 1-(1/n_1)
    m_1: float           #soil hydraulic parameter  [-]
    Ksat_1: float               #saturated hydraulic conductivity  [m/s]

    #SAND
    alpha_2: float
    theta_S2: float
    theta_R2: float
    n_2: float
    ##m_2 = 1-(1/n_2)
    m_2: float
    Ksat_2: float

    #Soil stress parameters
    theta_1_clay: float
    theta_2_clay: float

    theta_1_sand: float
    theta_2_sand: float

    # TREE PARAMETERS
    species:  str

    ###############################################################################
    # PHYSICAL CONSTANTS
    ###############################################################################
    Rho:  float  ##[kg m-3]
    g:  float # [m s-2]

    #ROOT PARAMETERS
    #diving by Rho*g since Richards equation is being solved in terms of \Phi (Pa)
    #Kr divided by rho*g
    Kr: float #soil-to-root radial conductance [m/sPa]
    qz: float
    #Ksax divided by rho*g                                       #unitless - parameter for the root mass distribution - Verma et al., 2014
    Ksax: float   #specific axial conductivity of roots  [ m/s]
    Aind_r: float                                       #m2 root xylem/m2 ground]

    #XYLEM PARAMETERS
    #kmax divided by rho*g
    kmax: float   #conductivity of xylem  [ m2/sPa]
    ap: float                                  #xylem cavitation parameter [Pa-1]
    bp: float                                #xylem cavitation parameter [Pa]
    Aind_x: float                           #m2 xylem/m2 ground]
    Phi_0: float                               #From bohrer et al 2005
    p: float                                          #From bohrer et al 2005
    sat_xylem: float                                #From bohrer et al 2005
    sapwood_area: float
    taper_top: float
    #TREE PARAMETERS
    Hspec: float                      #Height average of trees [m]
    LAI: float                       #[-] Leaf area index

    #########################################################################3
    #NHL PARAMETERS
    ###########################################################################

    scale_nhl:  float

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

    wp_s50:  float #value for oak from Mirfenderesgi
    c3:  float #value for oak from Mirfenderesgi

    LAD_norm:  str #LAD data


    #######################################################################
    #LEAF AREA DENSITY FORMULATION (LAD) [1/m]
    #######################################################################
    lad_scheme :  int  #0: default scheme, based on Lalic et al 2014; 1: scheme from NHL module

    #parameters if using penman-monteith transpiration scheme, based on Lalic et al 2014
    #if using NHL transpiration scheme, LAD is calculated in NHL module
    L_m: float  #maximum value of LAD a canopy layer
    z_m: float   #height in which L_m is found [m]

    ###########################################################################
    #PENMAN-MONTEITH EQUATION PARAMETERS
    ###########################################################################
    #W m^-2 is the same as J s^-1 m^-2
    #1J= 1 kg m2/s2
    #therefore 1W/m2 = kg/s3

    gb: float          #m/s Leaf boundary layer conductance
    Cp: float                # J/m3 K Heat capacity of air
    ga: float          #m/s Aerodynamic conductance
    lamb: float        #J/m3 latent heat of vaporization
    gama: float             #Pa/K psychrometric constant

    #########################################################################3
    #JARVIS PARAMETERS
    ###########################################################################

    gsmax: float      #m/s Maximum leaf stomatal conductance
    kr: float         #m2/W Jarvis radiation parameter
    kt: float       #K-2  Jarvis temperature parameter
    Topt: float           #K   Jarvis temperature parameter (optimum temperature)
    kd: float       #Pa-1 Jarvis vapor pressure deficit temperature
    hx50: float        #Pa  Jarvis leaf water potential parameter
    nl: float                   #[-] Jarvis leaf water potential parameter
    Emax: float        #m/s maximum nightime transpiration

#TODO convert to function
# Read configs from yaml file
logger.info("Reading config file" )

with open(config_file, "r") as yml_config:
    config_dict = yaml.safe_load(yml_config)

# Convert config dict to config dataclass
cfg = ConfigParams(**config_dict['model_options'], **config_dict['parameters'])