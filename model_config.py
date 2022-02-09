# Imports model configs from yaml file
import argparse
import yaml
from dataclasses import dataclass

# Command line argument for path of config file
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', nargs='?', default='model_config.yml')
args = parser.parse_args()
config_file = args.config_path

# Dataclass to hold the config parameters
@dataclass
class ConfigParams:
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

    #TREE PARAMETERS
    Hspec: float                      #Height average of trees [m]
    LAI: float                       #[-] Leaf area index
    Abasal: float         #[m2basal/m2-ground] xylem cross-sectional area and site surface ratio

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

# Read configs from yaml file
with open(config_file, "r") as yml_config:
    config_dict = yaml.safe_load(yml_config)

# Convert config dict to config dataclass
cfg = ConfigParams(**config_dict)