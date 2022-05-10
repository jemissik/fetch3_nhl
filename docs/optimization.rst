############
Optimization
############

***************
Getting Started
***************

Install requirements:

- Install FETCH3 requirements
- Install Ax: `Instructions for installing Ax <https://ax.dev/docs/installation.html>`_

.. note::
      Make sure to install Ax inside your fetch-dev environment

.. todo::
      Future version will move all optimization requirements to the fetch3 environment file
      for easier installation


Prepare optimizaion configuration file
--------------------------------------

See ``optimization_config_template.yml`` for an example of the optimization configuration file.

The optimization configuration file is similar to the FETCH3 configuration file.

To prepare this file:

1. Update the ``optimization_options`` section:

      - **obsfile**: path to the data file containing sapflux observations
      - **obsvar**: column header for the variable of interest in the observations file
      - **model_dir**: path to the fetch3 repository
      - **data_path**: path to the input data directory for the model
      - **output_dir**: directory where the optimization output will be written
      - **output_fname**: *Don't change.* Name of model output file
      - **experiment_name**: Name to label this optimization experiment with
      - **objective_name**: *Don't change for now - future version will have more options*. Objective function to use for optimization.
      - **ntrials**: number of trials to run in the optimization
2. Prepare the ``model_options`` section. This is identical to the model options section in the FETCH3 config file.
3. Prepare the ``parameters`` section. The parameters listed in this section are identical to those in the FETCH3 config file,
   but here you must specify additional information. For each parameter, you must specify:

      - **type**: fixed or range. Fixed parameters will not be optimized. Range parameters will be optimized.
      - **value or bounds**: For fixed parameters, you must specify the value of the parameter. For range parameters, you
        must specify bounds for the optimization.

.. todo::
      Will be updated soon:

      - cleaner options for specifying files and directories
      - additional options for objective functions


Running an optimization
-----------------------

To run an optimization, ``cd`` into the fetch3_nhl directory, then run::

      python optimization_run.py  --config_path your_config_path

Outputs will be saved in a folder inside the ``working_dir`` you specified, labeled with the experiment name and timestamp
of the optimization run.

.. note::
    Replace the paths and filenames in this example with the actual paths and files you are using.

See ``optimization_results.ipynb`` for an example of how to explore the optimization results.

.. ***************************
.. Optimization code reference
.. ***************************

.. .. todo::

..       This page is a work in progress. More detailed instructions and an updated
..       code reference will be added soon.
