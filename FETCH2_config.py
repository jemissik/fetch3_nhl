from pathlib import Path
import pandas as pd

fparams = {}
#INPUT DATA
fparams['BASE'] = Path.cwd()
fparams['DATA'] = "Derek_data_test30Min.csv"

fparams['start_time'] = pd.to_datetime('2007-01-01 00:00:00') #begining of simulation
fparams['end_time'] = pd.to_datetime('2007-01-01 00:30:00')   #end of simulation

#############################################################################
#MODEL PARAMETERS
#Values according to Verma et al., 2014

############################################################################

params={}

#OTHER PARAMETERS #TODO change these to constants
params['Rho']=1000       #[kg/m3]
params['g']=9.8          #[m/s2]

#SOIL PARAMETERS - USING VAN GENUCHTEN RELATIONSHIPS

#CLAY
params['alpha_1']=0.8                        #soil hydraulic parameter [1/m]
params['theta_S1']=0.55                      #saturated volumetric soil moisture content [-]
params['theta_R1']=0.068                     #residual volumetric soil moisture content [-]
params['n_1']=1.5                            #soil hydraulic parameter  [-]
params['m_1']=1-(1/params['n_1'])            #soil hydraulic parameter  [-]
params['Ksat_1']=1.94*10**(-7)               #saturated hydraulic conductivity  [m/s]

#SAND
params['alpha_2']=14.5
params['theta_S2']=0.47
params['theta_R2']=0.045
params['n_2']=2.4
params['m_2']=1-(1/params['n_2'])
params['Ksat_2']=3.45*10**(-5)

#Soil stress parameters
theta_1_clay=0.08
theta_2_clay=0.12

theta_1_sand=0.05
theta_2_sand=0.09


#ROOT PARAMETERS
#diving by Rho*g since Richards equation is being solved in terms of \Phi (Pa)
params['Kr']=(7.2*10**(-10))/(params['Rho']*params['g']) #soil-to-root radial conductance [m/sPa]
params['qz']=9                                           #unitless - parameter for the root mass distribution - Verma et al., 2014
params['Ksax']=(10**(-5))/(params['Rho']*params['g'])    #specific axial conductivity of roots  [ m/s]
params['Aind_r']=1                                       #m2 root xylem/m2 ground]


#XYLEM PARAMETERS
params['kmax']=(10**(-5))/(params['Rho']*params['g'])    #conductivity of xylem  [ m2/sPa]
params['ap']=2*10**(-6)                                  #xylem cavitation parameter [Pa-1]
params['bp']=-1.5*10**(6)                                #xylem cavitation parameter [Pa]
params['Aind_x']=8.62*10**(-4)                           #m2 xylem/m2 ground]
params['Phi_0']=5.74*10**8                               #From bohrer et al 2005
params['p']=20                                           #From bohrer et al 2005
params['sat_xylem']=0.573                                #From bohrer et al 2005

#TREE PARAMETERS
params['Hspec']=14                      #Height average of trees [m]
params['LAI']=1.5                       #[-] Leaf area index
params['Abasal']=8.62*10**(-4)          #[m2basal/m2-ground] xylem cross-sectional area and site surface ratio
#########################################################################3
#PENMAN-MONTEITH EQUATION PARAMETERS
###########################################################################
#W m^-2 is the same as J s^-1 m^-2
#1J= 1 kg m2/s2
#therefore 1W/m2 = kg/s3

kr=5*10**(-3)         #m2/W Jarvis radiation parameter
kt=1.6*10**(-3)       #K-2  Jarvis temperature parameter
Topt=289.15           #K   Jarvis temperature parameter (optimum temperature)
kd=1.1*10**(-3)       #Pa-1 Jarvis vapor pressure deficit temperature
hx50=-1274000         #Pa  Jarvis leaf water potential parameter



nl=2                   #[-] Jarvis leaf water potential parameter
gsmax=10*10**(-3)      #m/s Maximum leaf stomatal conductance
gb=2*10**(-2)          #m/s Leaf boundary layer conductance
LAI=1.5                #[-] Leaf area index
Cp=1200                # J/m3 K Heat capacity of air
ga=2*10**(-2)          #m/s Aerodynamic conductance
lamb=2.51*10**9        #J/m3 latent heat of vaporization
gama=66.7              #Pa/K psuchrometric constant
Emax=1*10**(-9)        #m/s maximum nightime transpiration
