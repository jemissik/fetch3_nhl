*********************************
Prepare input files for the model
*********************************

Meteorological data
===================

Prepare a .csv file with the required variables. See ``FLX_US-UMB_FLUXNET2015_SUBSET_HH_2007-2017_beta-4.csv`` in the
data folder for an example, although your file doesn't need to have all of the variables included in this file.

By default, the column headers in the input data are assumed to match the Ameriflux standard variable names. If
different column headers are used, the mapping of the column headers to the required variables but be specified in the
config file.

The meteorological variables that are required depend on whether you are using PM or NHL transpiration.


**If using the PM transpiration scheme, the file must include:**

- *TIMESTAMP_START*: Timestamp (in local standard time) using either Ameriflux format (YYYYMMDDHHMM) or standard format
  (YYYY-MM-DD HH:MM)
- *P_F*: Precipitation [mm]
- *TA_F*: Air temperature [deg C]
- *SW_IN_F*: Incoming shortwave radiation [W m-2]
- *VPD_F*: Vapor pressure deficit [hPa]


**If using the NHL transpiration scheme, the file must include:**

- *TIMESTAMP_START*: Timestamp (in local standard time) using either Ameriflux format (YYYYMMDDHHMM) or standard format
  (YYYY-MM-DD HH:MM)
- *P_F*: Precipitation [mm]
- *TA_F*: Air temperature [deg C]
- *SW_IN_F*: Incoming shortwave radiation [W m-2]
- *VPD_F*: Vapor pressure deficit [hPa]
- *WS_F*: Wind speed above the canopy [m s-1]
- *USTAR*: Friction velocity [m s-1]
- *PPFD_IN*: Incoming photosynthetic photon flux density (PAR) [µmolPhoton m-2 s-1]
- *CO2_F*: CO2 concentration [µmolCO2 mol-1]
- *PA_F*: Atmospheric pressure [kPa]

.. important::
    - If different column headers are used, this must be specified in the configuration file. See
      :ref:`Model Configuration`
    - The data should be gap-filled.


Leaf area density profile
=========================

If using the NHL transpiration scheme, you must also include a .csv file with the normalized leaf
area density profile.

This file should include:

- *z_h*: The normalized height for each layer (i.e. the height of the
  canopy layer z divided by the tree height h)
- Normalized LAD of each layer. You can include data for more than one species in
  this file, and each column should be labeled with the species name or abbreviation.

See ``LAD_data.csv`` for an example.


If using the PM transpiration scheme, the LAD profile will be calculated using a builtin function,
using parameters specified in the config file.

****************************************
Prepare configuration file for the model
****************************************

Model setup options and model parameters are read from a .yml file.

See :ref:`Model Configuration` for instructions about preparing this file.