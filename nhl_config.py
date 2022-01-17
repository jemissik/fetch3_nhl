from model_config import dt0, dz, start_time, end_time, input_fname, dt, Hspec, LAI

species = "ES"

height_sp = Hspec
total_LAI_sp = LAI

Cd = 0.2 # Drag coefficient
alpha_ml = 0.1  # Mixing length constant
mean_crown_area_sp = 17.02
total_crown_area_sp = 83614.7803393742
plot_area = 75649.5511
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

LAD_norm = 'LAD_data.csv' #LAD data
met_data = input_fname
met_dt = dt
