*********************************
Prepare input files for the model
*********************************

Meteorological data
===================

Prepare a .csv file with the required variables. See ``UMBS_flux_2011.csv`` in the data folder for an example,
although your file doesn't need to have all of the variables that are included in this file.

The meteorological variables that are required depend on whether you are using PM or NHL transpiration.

.. note::
    - The column headers in the .csv file must match the names used below. Most of these names/units
      match those used by AmeriFlux, except for Timestamp and VPD_kPa
    - The data should be gap-filled.

**If using the PM transpiration scheme, the file must include:**

- *Timestamp*
- *P_F*: Precipitation [mm]
- *TA_F*: Air temperature [deg C]
- *SW_IN_F*: Incoming shortwave radiation [W m-2]
- *VPD_kPa*: Vapor pressure deficit [kPa]


**If using the NHL transpiration scheme, the file must include:**

- *Timestamp*
- *P_F*: Precipitation [mm]
- *TA_F*: Air temperature [deg C]
- *SW_IN_F*: Incoming shortwave radiation [W m-2]
- *VPD_kPa*: Vapor pressure deficit [kPa]
- *WS_F*: Wind speed above the canopy [m s-1]
- *USTAR*: Friction velocity [m s-1]
- *PPFD_IN*: Incoming photosynthetic photon flux density (PAR) [µmolPhoton m-2 s-1]
- *CO2_F*: CO2 concentration [µmolCO2 mol-1]
- *RH*: Relative humidity [%]
- *PA_F*: Atmospheric pressure [kPa]

.. todo::
    A future version will allow the user to specify different column names for the required met data

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