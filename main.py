import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

from NHL_functions import *

# Example script to run model for the 2011 test data

####PARAMETERS#####
Cd = 0.1  # TODO Drag coefficient
alpha = 0.1  # TODO Mixing length constant

###Read in and format the data

# Read in the .mat data
mat = scipy.io.loadmat('data/2011_data.mat', squeeze_me =True)

species_list = list(mat['fn']) # list of species
height_sp = dict(zip(species_list, mat['Hspec'])) # vector, mean tree height for each species [m]
mean_crown_area_sp = dict(zip(species_list, mat['CrownArea'])) # vector, mean crown area for each species [m2]
total_crown_area_sp = dict(zip(species_list, mat['TotalCrownSP'])) # vector, total crown area for each species [m2]
dz = mat['dz'][0] # Vertical discretization length [m]
plot_area = mat['PLotArea'][0] # Plot area [m2]
total_LAI_sp = dict(zip(species_list, mat['TotLAI'])) # vector, total leaf area index for each species [m2-leaf/m2-ground]
t = mat['t'] # time in seconds
t0 = mat['t0'][0] # 5am time in seconds (18,000 seconds, from simulation start at 00:00 AM). Changed to one value instead of vector
latitude = mat['LAT'] # latitude of site [degrees]
longitude = mat['LONG'] # longitude of site [degrees]

# Convert the met data to a dataframe
met_data = pd.DataFrame(mat['DCRU'], columns = ['DOY', 'CO2', 'RH', 'Ustar', 'U_top', 'Ta_top', 'PAR', 'Press'])
met_data['Time'] = (t + t0)/3600  # TODO What is this time shift for?

# Reformat model parameters
# Columns : Vcmax, alpha (fitting parameter of gs model), alpha_p (ratio of horizontal to vertical projections of leaves)
# One row per species
params = pd.DataFrame(np.reshape(mat['params'], (-1, 3)), index = species_list, columns = ['Vcmax25', 'alpha', 'alpha_p'])

# LAD data
LAD = pd.DataFrame(mat['LAD'])
LAD.columns = ['z_h'] + species_list

# TODO - Overwriting the LAI data in the input file
total_LAI_sp_orig = total_LAI_sp.copy()
total_LAI_sp = np.array([1.1,1.45,0.84,0.044])*1.176*1.1 # vector, total leaf area index for each species [m2-leaf/m2-ground]

# TODO bs factor - will need to be altered. Overwriting the actual total crown area for each species
crown_scaling = np.array([2, 0.2, 0.1, 8])
total_crown_area_sp_orig = total_crown_area_sp.copy()
total_crown_area_sp = total_LAI_sp * crown_scaling / sum(total_LAI_sp * crown_scaling) * plot_area
total_LAI_sp = dict(zip(species_list, total_LAI_sp))
total_crown_area_sp = dict(zip(species_list, total_crown_area_sp))

# Run model for each species
#for species in species_list:
species = "ES"
# Extract parameters
Vcmax25 = params.loc[species, 'Vcmax25']
alpha_gs = params.loc[species, 'alpha']
alpha_p = params.loc[species, 'alpha_p']

#TODO temp - can just pass in directly to NHL
total_LAI_spn = total_LAI_sp[species]
total_crown_area_spn = total_crown_area_sp[species]
mean_crown_area_spn = mean_crown_area_sp[species]

i = 0
NHL_trans_sp_stem, NHL_tot_trans_sp_tree, NHL_trans_sp_crownarea, NHL_trans_sp_groundarea = calc_NHL(
    dz, height_sp[species], Cd, met_data.U_top[i], met_data.PAR[i], met_data.CO2[i], Vcmax25, alpha_p,
    total_LAI_spn, plot_area, total_crown_area_spn, mean_crown_area_spn, LAD[species], LAD.z_h,
    met_data.RH[i], met_data.Ta_top[i], met_data.Press[i], doy = met_data.DOY[i], lat = latitude,
    long = longitude, time_offset = -5, time_of_day = met_data.Time[i]) #TODO need to fix time!
