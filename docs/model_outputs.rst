#############
Model outputs
#############

The following files will be created in the output directory:

- ``fetch3.log``: Log file for the model run
- outputs from the NHL module:
    - nhl_zenith.csv: Timeseries of calculated zenith angle
    - nhl_out.nc: NHL transpiration
    - nhl_out_modelres: NHL trasnpiration interpolated to the resolution used by FETCH3
- FETCH3 outputs:
    - ds_canopy.nc: Dataset with the canopy variables:
        - S_kx: [unitless] Cavitation of stem xylem, given by eqn S.79 in Silva et al 2022
        - trans_2d: [m3H2O m-2crown_projection m-1stem s-1] Transpiration
        - H: [MPa] Water potential
        - K: [m s-1] Hydraulic conductivity
        - Capac: [Pa-1] Capacitance
    - ds_root.nc: Dataset with the root variables:
        - Kr_sink [1/sPa]: Effective root radial conductivity
        - S_kr [unitless]: Cavitation of root xylem, given by eqn S.77 in Silva et al 2022
        - S_sink [unitless]: Feddes root water uptake stress function, given by equations S.73, 74 and 75 in Silva et al 2022
        - EVsink_ts [m3H2O m-2ground m-1depth s-1]: Root water uptake
        - H: [MPa] Water potential
        - K: [m s-1] Hydraulic conductivity
        - Capac: [Pa-1] Capacitance
    - ds_soil.nc: Dataset with the soil variables:
        - THETA: [m3 m-3] Volumetric water content
        - H: [MPa] Water potential
        - K: [m s-1] Hydraulic conductivity
        - Capac: [Pa-1] Capacitance
    - ds_all.nc: Dataset with H, Capac, and K concatenated for the entire z array
    (these variables are also written in ds_canopy, ds_root, and ds_soil, so in most
    cases it is easier to use those datasets instead)