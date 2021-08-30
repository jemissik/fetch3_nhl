import numpy as np
import scipy.io
import pandas as pd 

# Example script to run model for the 2011 test data  

###Read in and format the data 

# Read in the .mat data 
mat = scipy.io.loadmat('data/2011_data.mat', squeeze_me =True)

species = mat['fn'] # list of species 
height_sp = mat['Hspec'] # vector, mean tree height for each species [m]
mean_crown_area_sp = mat['CrownArea'] # vector, mean crown area for each species [m2]
total_crown_area_sp = mat['TotalCrownSP'] # vector, total crown area for each species [m2]
dz = mat['dz'] # Vertical discretization length [m]
plot_area = mat['PLotArea'] # Plot area [m2]
total_LAI_sp = mat['TotLAI'] # vector, total leaf area for each species [m2-leaf/m2-ground]
t = mat['t'] # time in seconds 
t0 = mat['t0'][0] # 5am time in seconds (18,000 seconds, from simulation start at 00:00 AM). Changed to one value instead of vector 
latitude = mat['LAT'] # latitude of site [degrees]
longitude = mat['LONG'] # longitude of site [degrees]

# Convert the met data to a dataframe 
met_data = pd.DataFrame(mat['DCRU'], columns = ['DOY', 'CO2', 'RH', 'Ustar', 'U_top', 'Ta_top', 'PAR', 'Press'])

# Reformat model parameters
# Columns : Vcmax, alpha (fitting parameter of gs model), alpha_p (ratio of horizontal to vertical projections of leaves)
# One row per species 
params = pd.DataFrame(np.reshape(mat['params'], (-1, 3)), index = species, columns = ['Vcmax', 'alpha', 'alpha_p'])

# LAD data 
LAD_data = pd.DataFrame(mat['LAD'])
LAD_data.drop(4, axis = 1, inplace = True) # drop the extra column of LAD data... there were 5 columns but 4 species
LAD_data.columns = mat['fn']
