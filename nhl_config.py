#
species = 'ES'

#Replicated from FETCH3 config
dt0 = 20  #model temporal resolution [s]
dz = 0.1  #model spatial resolution [m]
start_time = "2011-05-01 10:00:00" #begining of simulation
end_time = "2011-05-01 13:30:00" #end

#in FETCH3 (but values match UMBS)
height_sp = 22
total_LAI_sp = 1.5 #from FETCH3

#Defaults from "ES" in MATLAB test data

Cd = 0.2 # Drag coefficient
alpha_ml = 0.1  # Mixing length constant
mean_crown_area_sp = 17.02
total_crown_area_sp = 83614.7803393742
plot_area = 75649.5511
# total_LAI_sp = 1.1 * 1.176 * 1.1 # Value that was written by matlab script, doesn't match data files
# crown_scaling = 2
latitude = 39.9137
longitude = -74.596
time_offset = -5
Vcmax25 = 31.15
alpha_gs = 7.3200
alpha_p = 1
Cf = 0.85  #Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
x = 1  #Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

wp_s50 = -9.1 * 10**5 #value for oak from Mirfenderesgi
c3 = 12.3 #value for oak from Mirfenderesgi

LAD_norm = 'LAD_data.csv'
met_data = 'UMBS_flux_2011.csv'
met_dt = 1800 #Half hourly data
