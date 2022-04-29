###############
Getting Started
###############

To get started with FETCH3:

1. Install python, FETCH3, and its dependencies
2. Prepare input data for the model
3. Prepare configuration file

************
Installation
************

Install Python
==============

To use FETCH3, you must first have Python and the conda package manager
installed. There are two options for this:

- **Install Anaconda**: This is the recommended option for those who are new to
  Python. Anaconda comes with the Spyder IDE, which provides an interface similar to
  RStudio and MATLAB, and Jupyter Notebook, which can be used to run interactive Python
  notebooks such as those in FETCH3's optimization examples. It also includes a graphical
  interface (Anaconda Navigator) for launching applications and managing conda environments
  without using the command line. To install Anaconda, see
  `directions for installing Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_.
- **Install Miniconda**: If you want a more minimal installation without any extra
  packages, would prefer to handle installing a Python IDE yourself, and would prefer
  to work from the command line instead of using the graphical interface provided
  by Anaconda Navigator, you might prefer to install Miniconda rather than Anaconda.
  Miniconda is a minimal version of Anaconda that includes just conda, its dependencies,
  and Python. To install Miniconda, see
  `directions for installing Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Install FETCH3
==============

Clone the FETCH3 repository from `FETCH3's GitHub page <https://github.com/jemissik/fetch3_nhl>`_.

.. note::
    The NHL transpiration code is in a submodule inside the FETCH3 repository. Make sure
    you have cloned the files in the `nhl_transpiration` submodule!

.. todo::
    - The NHL transpiration code will be moved out of a submodule in a future version
    - Eventually FETCH3 will be released as a package to make installation simpler

Install FETCH3's dependencies
=============================

It is recommended to create a new conda environment for FETCH3, using the provided environment file.

**To install using the command line**:

1. Use ``cd`` to navigate into the FETCH3 directory that you cloned from GitHub.
2. Create FETCH3's conda environment::

    conda env create --file fetch3_requirements.yml

3. To activate the conda environment, run::

    conda activate fetch3-dev

  Make sure you're using the FETCH3 conda environment when you try to run the model.

  See this `cheat sheet for working with conda <https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf>`_ for
  a helpful list of conda commands.

.. note::
    For Windows users, use the **Anaconda Prompt** application that was installed with Anaconda Navigator
    to run these commands. See `Instructions for Anaconda Prompt <https://docs.anaconda.com/anaconda/user-guide/getting-started/#cli-hello>`_

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
- *TA_F*: Air temperature [deg C]
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

*****************
Running the model
*****************

Setting input and output directories
====================================

The input data files, config file, and output directory can all be in locations of your
choice, and these locations are specified as command line arguments when you run the model.
If they aren't specified, defaults will be used.

**Default input and output directories:**

* Input meteorological data: ``./data/``
* Input configuration file: ``./model_config.yml``
* Model output and logs: ``./output/``
  If using the default output directory, a directory ``./output/`` will be created
  if it doesn't already exist.

Running the model from the command line
========================================

Run the model by running ``main.py``

To specify an input config file, data directory, or output directory in a location other than the
default, different directories can be specified as command line arguments, for example::
      python main.py --config_path /Users/username/fetch3/user_model_config.yml
      --data_path /Users/username/fetch3/user_data/ --output_path /Users/username/fetch3/output/

.. note::
    Replace the paths and filenames in this example with the actual paths and files you are using.

If the arguments ``--config_path``, ``--data_path``, and ``--output_path`` are omitted when running the
model from the command line, the defaults will be used.
