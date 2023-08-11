"""
###################
Model configuration
###################

Reads in model configuration from .yml file

Model setup options and model parameters are read from a .yml file, which can be
modified by the user.

See :ref:`scaling` for details about how parameters are converted.


*************************
Modifying the config file
*************************

.yml file contents
==================
See ``model_config.yml`` for an example.

.. note::
    FETCH3 can also use the config file format that is used for optimization runs, as long as ``value`` is specified for
    every parameter.

Model options
-------------
* ``make_experiment_dir`` (bool, optional): Whether or not FETCH should create a new experiment inside the specified output directory
  for the results of the model run. If True, FETCH will write the model outputs to a new directory labeled with the
  experiment name and the timestamp of the run. If False or not provided, the outputs will be written directly inside the
  output directory.
* ``experiment_name`` (str, optional): Optional label for the run. This is simply used for labeling the output directory.
* ``input_fname`` (str): File for input met data
* ``met_column_labels`` (dict): Dictionary specifying the mapping of the column headers in your input file to the
  required input variables. This is needed if your column headers differ from the default variable names. See
  :ref:`Prepare input files for the model` for a list of the default variable names. Each element in the dictionary
  should be formatted as <your column header>: <standard variable name>, for example::

    met_column_labels:
      CO2_F_MDS: CO2_F

  Alternately, the dictionary can also be formatted as::

    met_column_labels: {'CO2_F_MDS': 'CO2_F'}


* ``start_time`` (str): ["YYYY-MM-DD HH:MM:SS"] Begining of simulation
* ``end_time`` (str): ["YYYY-MM-DD HH:MM:SS"] End of simulation

* ``dt``(int): [seconds] Input data resolution
* ``tmin`` (int): [s]  #tmin #TODO

Site information
----------------
* ``latitude`` (float): Latitude of site in decimal degrees
* ``longitude`` (longitude): Longitude of site in decimal degrees
* ``time_offset`` (float): Offset from UTC time, e.g EST = UTC -5 hrs. This is used in the calculation of the zenith angle
  to figure out the standard meridian of your location. It is not used to shift the timestamps in your input data. Timestamps
  in the input data should be in local standard time.

Run options - printing
----------------------
Options to turn printing off or specify print frequency. Printing the run
progress every timestep will slow down the model run (and make your log file very
long). The model will run faster if printing is turned off or set to print more
infrequently.

* ``print_run_progress`` (bool): Turns on/off printing for progress of the model run.
  ``print_run_progress: False`` will turn off printing the progress of the timesteps
  calculated.
* ``print_freq`` (int): Interval of timesteps to print if ``print_run_progress = True``
  (e.g. ``print_freq: 100`` will print every 100 timesteps).

Transpiration options
---------------------

* ``transpiration_scheme`` (int): Whether to use the PM transpiration scheme or the NHL transpiration scheme

  * 0: PM transpiration
  * 1: NHL transpiration

* ``lad_scheme`` (int) : 0: default scheme, based on Lalic et al 2014; 1: scheme from NHL module
* ``LAD_norm`` (str):  File with normalized LAD data

* ``sum_LAI_plot`` (float): Total LAI for the plot
* ``Cd`` (float): Drag coefficient
* ``alpha_ml`` (float): Mixing length constant


Numerical solution time and space constants (dz and dt0)
--------------------------------------------------------
The finite difference discretization constants

* ``dt0`` (int): model temporal resolution [s]
* ``dz`` (float): model spatial resolution [m]
* ``stop_tol`` (float): stop tolerance of equation converging


Soil boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^
Here the user can choose the desired soil boundary conditions as described below.
The configuration used in the example config file follows Verma et al. 2014.

* ``UpperBC`` (int): Upper boundary condition
  Options:
    * 1: no flux (Neuman)
    * 0: infiltration
* ``BottomBC`` (int): Bottom boundary condition
  Options:
    * 2: free drainage
    * 1: no flux (Neuman)
    * 0: constant potential (Dirichlet)

* ``frac_infiltration`` (float): Fraction of precipitation that should infiltrate to
  the soil (can use to account for interception & runoff)

Site parameters
----------------

Soil spatial discretization
^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``Soil_depth`` (float): *[m]* depth of soil column

Configuration of soil duplex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Depths of layer/clay interface

* ``sand_d`` (float): # Depth of sand layer #m
* ``clay_d`` (float): Depth of clay layer. #m

Soil initial conditions
^^^^^^^^^^^^^^^^^^^^^^^
* ``initial_swc_clay`` (float): initial soil water content for the clay layer [m3 m-3]
* ``initial_swc_sand`` (float): initial soil water content for the sand layer [m3 m-3]
* ``soil_moisture_bottom_boundary`` (float): Soil moisture content [m3 m-3] for bottom boundary. Only used if the Dirichlet
  bottom boundary condition is used

Soil parameters - using Van Genuchten relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clay:

* ``alpha_1`` (float): soil hydraulic parameter *[1/m]*
* ``theta_S1`` (float): saturated volumetric soil moisture content *[-]*
* ``theta_R1`` (float): residual volumetric soil moisture content *[-]*
* ``n_1`` (float): soil hydraulic parameter *[-]*
* ``Ksat_1`` (float): saturated hydraulic conductivity *[m/s]*

Sand: same definitions as above, but for sand

* ``alpha_2`` (float):
* ``theta_S2`` (float):
* ``theta_R2`` (float):
* ``n_2`` (float):
* ``Ksat_2`` (float):



Species parameters
^^^^^^^^^^^^^^^^^^

Each species will have a section in the config file, labeled with the species name,
that must include these parameters.

Soil stress parameters:

* ``theta_1_clay`` (float)
* ``theta_2_clay`` (float)
* ``theta_1_sand`` (float)
* ``theta_2_sand`` (float)

Root parameters
^^^^^^^^^^^^^^^
* ``Root_depth`` (float): Depth of root column
* ``Kr`` (float): soil-to-root radial conductance *[m/sPa]*
* ``qz`` (float): unitless - parameter for the root mass distribution - Verma et al., 2014
* ``Ksax`` (float): specific axial conductivity of roots *[m/s]*
* ``Aind_r`` (float): *[m2 root xylem/m2 ground]*

Xylem parameters
^^^^^^^^^^^^^^^^
* ``kmax`` (float): conductivity of xylem *[m2/sPa]*
* ``ap`` (float): xylem cavitation parameter *[Pa-1]*
* ``bp`` (float): xylem cavitation parameter *[Pa]*
* ``Phi_0`` (float): From bohrer et al 2005
* ``p`` (float): From bohrer et al 2005
* ``sat_xylem`` (float): From bohrer et al 2005

Tree parameters
^^^^^^^^^^^^^^^
* ``sapwood_depth``: [cm]
* ``Hspec`` (float): Height average of trees *[m]*
* ``LAI`` (float): *[-]* Leaf area index
* ``dbh``: [cm]
* ``stand_density_sp``: species-specific stand density [trees ha-1]

NHL transpiration scheme parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If using the NHL transpiration scheme.

* ``scale_nhl`` (float): scaling parameter for NHL transpiration
* ``mean_crown_area_sp`` (float):


* ``Cf`` (float): Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
* ``x`` (float): Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

* ``Vcmax25`` (float):
* ``alpha_gs`` (float):
* ``alpha_p`` (float):

* ``wp_s50`` (float):
* ``c3`` (float):


Penman-Monteith transpiration parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


parameters if using penman-monteith transpiration scheme, based on Lalic et al 2014
if using NHL transpiration scheme, LAD is calculated in NHL module

* ``L_m`` (float): maximum value of LAD a canopy layer
* ``z_m`` (float): height in which L_m is found [m]


Penman-Monteith equation parameters
Note: W m^-2 is the same as J s^-1 m^-2
1J= 1 kg m2/s2
therefore 1W/m2 = kg/s3

* ``gb`` (float): *[m/s]* Leaf boundary layer conductance
* ``Cp`` (float): *[J/m3 K]* Heat capacity of air
* ``ga`` (float): *[m/s]* Aerodynamic conductance
* ``lamb`` (float): *[J/m3]* latent heat of vaporization
* ``gama`` (float): *[Pa/K]* psychrometric constant

Jarvis parameters

* ``gsmax`` (float): *[m/s]* Maximum leaf stomatal conductance
* ``kr`` (float): *[m2/W]* Jarvis radiation parameter
* ``kt`` (float): *[K-2]* Jarvis temperature parameter
* ``Topt`` (float): *[K]* Jarvis temperature parameter (optimum temperature)
* ``kd`` (float): *[Pa-1] Jarvis vapor pressure deficit temperature
* ``hx50`` (float): *[Pa]*  Jarvis leaf water potential parameter
* ``nl`` (float): *[-]* Jarvis leaf water potential parameter
* ``Emax`` (float): *[m/s]* maximum nightime transpiration

"""
from __future__ import annotations

import collections
import logging
from copy import deepcopy
from os import PathLike

from attrs import define, field, fields
from typing import ClassVar, Optional
from enum import IntEnum
import yaml

from fetch3.scaling import calc_Aind_x, calc_LAIc_sp, calc_xylem_cross_sectional_area
from fetch3.utils import load_yaml, deprecation

logger = logging.getLogger(__file__)


class TranspirationScheme(IntEnum):
    PM = 0
    pm = 0
    Penman_Monteith = 0
    penman_monteith = 0
    NHL = 1
    nhl = 1


def get_enum(val, enum):
    """Get enum from string or int, normalize string, returning enum value"""
    if isinstance(val, str):
        val = val.lower().replace("-", "_")
        return getattr(enum, val)
    elif isinstance(val, int):
        return enum(val)


@define
class ModelOptions:
    # File for input met data
    input_fname: str

    start_time: str  # begining of simulation
    end_time: str  # end

    dt: int  # seconds - input data resolution
    tmin: int  # tmin [s]

    latitude: float
    longitude: float
    time_offset: float  # Offset from UTC time, e.g EST = UTC -5 hrs

    # Printing slows down model run
    # Options to turn printing off or specify print frequency
    print_run_progress: bool  # Turn on/off printing for progress of time steps calculated
    print_freq: int  # Interval of timesteps to print if print_run_progress = True (e.g. 1 will print every time step)


    lad_scheme: int  # 0: default scheme, based on Lalic et al 2014; 1: scheme from NHL module

    ###############################################################################
    # NUMERICAL SOLUTION TIME AND SPACE CONSTANTS (dz and dt0)
    ###############################################################################
    # The finite difference discretization constants
    dt0: int  # model temporal resolution [s]
    dz: float  # model spatial resolution [m]

    stop_tol: float  # stop tolerance of equation converging

    # TREE SPECIES PARAMETERS
    species: str

    # CONFIGURING SOIL BOUNDARY CONDITIONS
    # Here the user can choose the desired contition by setting the numbers as
    # described below

    # The configuration used follows Verma et al. 2014

    #############################################################################

    # Upper Boundary condition

    # 1 = no flux (Neuman)
    # 0 = infiltration

    # Bottom Boundary condition

    # 2 = free drainage
    # 1 = no flux (Neuman)
    # 0 = constant potential (Dirichlet)

    UpperBC: int
    BottomBC: int
    LAD_norm: str = None  # LAD data

    make_experiment_dir: bool = False
    experiment_name: str = None
    met_column_labels: dict = None


@define
class NHLModelOptions(ModelOptions):
    """NHL Model Options"""
    zenith_method: str = "CN"  # "CN" or "fetch2" zenith angle calculation method


@define
class BaseParameters:
    g: ClassVar = 9.81  # Gravitational acceleration [m/s2]
    Rho: ClassVar = 998  # Density of water [kg/m3]

    #############################################################################
    # MODEL PARAMETERS
    # Values according to Verma et al., 2014
    ############################################################################


    # SOIL SPATIAL DISCRETIZATION

    Root_depth: float  # [m] depth of root column
    Soil_depth: float  # [m]depth of soil column

    ####################################################################
    # CONFIGURATION OF SOIL DUPLEX
    # depths of layer/clay interface
    #####################################################################

    sand_d: float  # 4.2----top soil #m
    clay_d: float  # 0------4.2 #m

    # SOIL INITIAL CONDITIONS
    initial_swc_clay: float  # [m3 m-3]
    initial_swc_sand: float  # [m3 m-3]

    soil_moisture_bottom_boundary: None  # Soil moisture for bottom boundary condition (if using Dirichlet boundary)
    # SOIL PARAMETERS - USING VAN GENUCHTEN RELATIONSHIPS

    # CLAY
    alpha_1: float  # soil hydraulic parameter [1/m]
    theta_S1: float  # saturated volumetric soil moisture content [-]
    theta_R1: float  # residual volumetric soil moisture content [-]
    n_1: float  # soil hydraulic parameter  [-]
    Ksat_1: float  # saturated hydraulic conductivity  [m/s]

    # SAND
    alpha_2: float
    theta_S2: float
    theta_R2: float
    n_2: float
    Ksat_2: float

    # Soil stress parameters
    theta_1_clay: float
    theta_2_clay: float

    theta_1_sand: float
    theta_2_sand: float

    # ROOT PARAMETERS
    Kr: float  # soil-to-root radial conductance [m/sPa]
    qz: float
    Ksax: float  # specific axial conductivity of roots  [ m/s]
    Aind_r: float  # m2 root xylem/m2 ground]

    # XYLEM PARAMETERS
    kmax: float  # conductivity of xylem  [ m2/sPa]
    ap: float  # xylem cavitation parameter [Pa-1]
    bp: float  # xylem cavitation parameter [Pa]
    Phi_0: float  # From bohrer et al 2005
    p: float  # From bohrer et al 2005
    sat_xylem: float  # From bohrer et al 2005

    taper_top: float

    sapwood_depth: float  # Sapwood depth [cm]
    dbh: float  # DBH [cm]
    stand_density_sp: float  # Species-specific stand density [trees ha-1]

    # TREE PARAMETERS
    Hspec: float  # Height average of trees [m]
    LAI: float  # [-] Leaf area index
    mean_crown_area_sp: float
    sum_LAI_plot: float


    # Infiltration
    frac_infiltration: float = 1

    m_1: float = field(init=False)
    m_2: float = field(init=False)
    sapwood_area: float = field(init=False)
    Aind_x: float = field(init=False)
    LAIc_sp: float = field(init=False)

    def __attrs_post_init__(self):

        # Calculate m_1 and m_2
        self.m_1 = 1 - (1 / self.n_1)
        self.m_2 = 1 - (1 / self.n_2)

        # divide Kr, Ksax, and kmax by rho*g
        # diving by Rho*g since Richards equation is being solved in terms of \Phi (Pa)
        self.Kr = self.Kr / (self.Rho * self.g)
        self.Ksax = self.Ksax / (self.Rho * self.g)
        self.kmax = self.kmax / (self.Rho * self.g)

        # Calculate sapwood area
        self.sapwood_area = calc_xylem_cross_sectional_area(self.dbh, self.sapwood_depth)

        # Calculate Aind_x
        self.Aind_x = calc_Aind_x(self.sapwood_area, self.mean_crown_area_sp)
        # Calculate LAIc_sp
        self.LAIc_sp = calc_LAIc_sp(self.LAI, self.mean_crown_area_sp, self.stand_density_sp)




@define
class PMParameters(BaseParameters):
    ###########################################################################
    # PENMAN-MONTEITH EQUATION PARAMETERS
    ###########################################################################

    # parameters if using penman-monteith transpiration scheme, based on Lalic et al 2014
    # if using NHL transpiration scheme, LAD is calculated in NHL module
    L_m: float = None  # maximum value of LAD a canopy layer
    z_m: float = None  # height in which L_m is found [m]
    # W m^-2 is the same as J s^-1 m^-2
    # 1J= 1 kg m2/s2
    # therefore 1W/m2 = kg/s3

    gb: float = None  # m/s Leaf boundary layer conductance
    Cp: float = None  # J/m3 K Heat capacity of air
    ga: float = None  # m/s Aerodynamic conductance
    lamb: float = None  # J/m3 latent heat of vaporization
    gama: float = None  # Pa/K psychrometric constant

    #########################################################################3
    # JARVIS PARAMETERS
    ###########################################################################

    gsmax: float = None  # m/s Maximum leaf stomatal conductance
    kr: float = None  # m2/W Jarvis radiation parameter
    kt: float = None  # K-2  Jarvis temperature parameter
    Topt: float = None  # K   Jarvis temperature parameter (optimum temperature)
    kd: float = None  # Pa-1 Jarvis vapor pressure deficit temperature
    hx50: float = None  # Pa  Jarvis leaf water potential parameter
    nl: float = None  # [-] Jarvis leaf water potential parameter
    Emax: float = None  # m/s maximum nightime transpiration


@define
class NHLParameters(BaseParameters):
    #########################################################################
    # NHL PARAMETERS
    #########################################################################

    scale_nhl: float = None

    Cd: float = None  # Drag coefficient
    alpha_ml: float = None  # Mixing length constant
    Cf: float = None  # Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
    x: float = None  # Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

    Vcmax25: float = None
    alpha_p: float = None

    # alpha_gs and m are the same parameter
    # old version of model used alpha_gs, so rename this parameter to m if it's in the config file
    alpha_gs: float = None
    m: float = None

    wp_s50: float = None  # value for oak from Mirfenderesgi
    c3: float = None  # value for oak from Mirfenderesgi

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # Rename alpha_gs to m if it's in the config file
        if self.alpha_gs is not None:
            self.m = self.alpha_gs
            self.alpha_gs = None


SCHEMES = {
    TranspirationScheme.PM: {"parameters": PMParameters, "model_options": ModelOptions},
    TranspirationScheme.NHL: {"parameters": NHLParameters, "model_options": NHLModelOptions},
}


# Dataclass to hold the config parameters
@define
class ConfigParams:
    transpiration_scheme: int | str | TranspirationScheme  # 0: PM transpiration; 1: NHL transpiration

    model_options: dict | ModelOptions | NHLModelOptions
    parameters: dict | BaseParameters | NHLParameters | PMParameters

    def __init__(self, model_options, parameters, transpiration_scheme=None):
        if transpiration_scheme is None:
            if "transpiration_scheme" not in model_options:
                raise ValueError("transpiration_scheme must be specified in model_options")
            transpiration_scheme = model_options.pop("transpiration_scheme")
        transpiration_scheme = get_enum(val=transpiration_scheme, enum=TranspirationScheme)

        if isinstance(model_options, dict):
            model_options = SCHEMES[transpiration_scheme]["model_options"](**model_options)
        if isinstance(parameters, dict):
            parameters = SCHEMES[transpiration_scheme]["parameters"](**parameters)
        self.__attrs_init__(transpiration_scheme=transpiration_scheme, model_options=model_options, parameters=parameters)

    @classmethod
    def from_deprecated_config(cls, config_path: Optional[str | PathLike] = None, config: Optional[dict] = None, species: Optional[str] = None):
        deprecation("Config format is deprecated in will be removed in a future version."
                    "Consult documentation for new format.")
        if config and config_path:
            raise ValueError("Only one of config and config_path can be specified")
        if config_path is not None:
            config = load_yaml(config_path)
        if species is None:
            species = list(config['species_parameters'].keys())[0]
            logger.info("No species was specified, so using species: " + species)
        # Check if the config file was the optimization config file format, and convert
        if "optimization_options" in list(config):
            site_param_dict = {}
            species_param_dict = {}
            for param in config["site_parameters"].keys():
                site_param_dict[param] = config["site_parameters"][param]["value"]
            for param in config["species_parameters"][species].keys():
                try:
                    species_param_dict[param] = config["species_parameters"][species][param]["value"]
                except KeyError as e:
                    logger.info(species, param)
                    logger.warning(repr(e))
                    raise
        else:
            site_param_dict = config["site_parameters"]
            species_param_dict = config["species_parameters"][species]

        return cls(**{"model_options": {**config["model_options"], 'species': species},
                      "parameters": {**site_param_dict, **species_param_dict}})

    @property
    def species(self):
        return self.model_options.species

    @property
    def Rho(self):
        return self.parameters.Rho

    @property
    def g(self):
        return self.parameters.g


def get_multi_config(config_path: Optional[str | PathLike] = None, config: Optional[dict] = None, species: Optional[str | list[str]] = None) -> list[ConfigParams]:
    """Get a list of ConfigParams objects from a config file or dict"""
    if config and config_path:
        raise ValueError("Only one of config and config_path can be specified")
    if config_path is not None:
        config = load_yaml(config_path)

    if "model_trees" in config:
        return config_from_groupers(config=config, species=species)
    elif "species_parameters" in config:
        species_list = species or list(config["species_parameters"].keys())
        return [ConfigParams.from_deprecated_config(config=config, species=species) for species in species_list]
    else:
        raise ValueError("Invalid config file format. Config file must contain either 'model_trees' key (current valid format)"
                         " or 'species_parameters' and 'site_parameters' keys (deprecated format)")


def get_single_config(config_path: Optional[str | PathLike] = None, config: Optional[dict] = None, species: Optional[str] = None) -> ConfigParams:
    """Get a list of ConfigParams objects from a config file or dict"""
    if config and config_path:
        raise ValueError("Only one of config and config_path can be specified")
    if config_path is not None:
        config = load_yaml(config_path)

    if "model_trees" in config:
        return config_from_groupers(config=config, species=species)[0]
    elif "species_parameters" in config:
        return ConfigParams.from_deprecated_config(config=config, species=species)
    else:
        raise ValueError("Invalid config file format. Config file must contain either 'model_trees' key (current valid format)"
                         " or 'species_parameters' and 'site_parameters' keys (deprecated format)")


def config_from_groupers(config, species: Optional[str | list[str]]  = None):
    if isinstance(species, str):
        species = [species]
    groups = config.get("groups")  # we do a .get b/c we don't want to raise an error if groups is not in config
    model_trees = config["model_trees"]  # this should always be in config, so we use a regular dict access
    configs = []
    for tree, parameters in model_trees.items():
        if species and tree not in species:
            continue
        model_options = deepcopy(config["model_options"])  # deepcopy b/c Config init modifies the dict
        parents = parameters.pop("parents", [])  # default empty list so we can iterate over it later

        parameter_check_dict = collections.defaultdict(list)
        for label, param_dict in zip(
                [tree, *parents],
                [parameters] + [groups[parent] for parent in parents]
        ):
            for param, value in param_dict.items():
                parameter_check_dict[param].append(label)
        for key, labels in parameter_check_dict.items():
            if len(labels) > 1:
                raise ValueError(f"Parameter {key} is defined more than once. It is defined in {labels} locations."
                                 "Overriding parameters from a parent in a child is not allowed.")

        for parent in parents:
            parameters.update(groups[parent])  # update the parameters dict with the parent parameters
        model_options["species"] = tree
        configs.append(ConfigParams(model_options=model_options, parameters=parameters))
    return configs


# Read configs from yaml file


# Convert config dict to config dataclass


def setup_config(config_file, species=None):
    logger.info("Reading config file")

    loaded_configs = load_yaml(config_file)

    if species is None:
        species = list(loaded_configs['species_parameters'].keys())[0]
        logger.info("No species was specified, so using species: " + species)
    # Check if the config file was the optimization config file format, and convert
    if "optimization_options" in list(loaded_configs):
        site_param_dict = {}
        species_param_dict = {}
        for param in loaded_configs["site_parameters"].keys():
            site_param_dict[param] = loaded_configs["site_parameters"][param]["value"]
        for param in loaded_configs["species_parameters"][species].keys():
            try:
                species_param_dict[param] = loaded_configs["species_parameters"][species][param]["value"]
            except KeyError as e:
                print(species, param)
                print(repr(e))
                raise
    else:
        site_param_dict = loaded_configs["site_parameters"]
        species_param_dict = loaded_configs["species_parameters"][species]

    cfg = ConfigParams(**{"model_options": {**loaded_configs["model_options"], 'species': species},
                          "parameters": {**site_param_dict, **species_param_dict}})
    return cfg


def save_calculated_params(fileout, cfg):
    with open(fileout, "w") as f:
        # Write model options from loaded config
        # Parameters for the trial from Ax
        yaml.dump(cfg, f)


if __name__ == "__main__":
    import pathlib
    _cp = pathlib.Path(__file__).parent.parent / "config_files/test_param_groups.yml"
    print(config_from_groupers(load_yaml(_cp))[0])
