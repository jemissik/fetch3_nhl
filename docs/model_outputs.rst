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
    - ds_all.nc
    - ds_canopy.nc
    - ds_root.nc
    - ds_soil.nc 