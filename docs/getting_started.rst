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
  If you have plans to modify the FETCH3 source code itself, see the :ref:`Developer guide` for instructions about
  forking the FETCH3 repository and working with git.

.. todo::
    - Eventually FETCH3 will be released as a package to make installation simpler

Install FETCH3's dependencies
=============================

It is recommended to create a new conda environment for FETCH3, using the provided environment file.

**To install using the command line**:

1. Use ``cd`` to navigate into the FETCH3 directory that you cloned from GitHub.
2. Create FETCH3's conda environment.

   For Linux, Windows, and ARM Macs::

     conda env create --file fetch3_requirements.yml

   For x86 Macs::

     conda env create --file fetch3_requirements_macx86.yml

   .. note::
      For Mac users:

      - Use ``fetch3_requirements.yml`` for M-series macs running a native version of conda
      - Use ``fetch3_requirements_macx86.yml`` for Intel Macs or M-series Macs running conda through Rosetta

3. To activate the conda environment, run::

    conda activate fetch3-dev

   Make sure you're using the FETCH3 conda environment when you try to run the model.

   See this `cheat sheet for working with conda <https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf>`_ for
   a helpful list of conda commands.

.. note::
    For Windows users, use the **Anaconda Prompt** application that was installed with Anaconda Navigator
    to run these commands. See `Instructions for Anaconda Prompt <https://docs.anaconda.com/anaconda/user-guide/getting-started/#cli-hello>`_


*********************
Test run of the model
*********************

Once everything is installed, try to run FETCH3 using the default test files that are installed with the model. This way,
you can make sure everything is working correctly before you move on to using your own data and configuration files.

To run the test case, use ``cd`` to navigate inside the directory with the FETCH3 code. To check that you're in the right place,
run ``ls`` and check that there is a file called ``main.py`` in the current directory, for example::


  >>> ls
  README.md                          optimization.ipynb
  __pycache__                        optimization_results.ipynb
  config_files                       optimization_run.py
  data                               output
  docs                               read_sapfluxnet_data.ipynb
  example_model_output.ipynb         run_debug.py
  explore_optimization_results.ipynb sapflux_test.ipynb
  fetch3                             scratch
  fetch3_requirements.yml            speed_test
  fetch3_requirements_m1.yml         test_output_slice.ipynb
  main.py                            umbs_census.ipynb

Once you're in the correct directory, you can run the model using the provided test files by running ``main.py``::

  python main.py

For runs using your own data and configuration file, you will specify the configuration file, data directory, and output directory
as command line arguments. Since we are ommitting these arguments, the default files will be used instead.

**Default input and output directories:**

* Input meteorological data: ``./data/``
* Input configuration file: ``./model_config.yml``
* Model output and logs: ``./output/``
  If using the default output directory, a directory ``./output/`` will be created
  if it doesn't already exist.

If this test case runs successfully, you can move on to preparing your own data and configuration files.
If you have errors, see the :ref:`Troubleshooting` section.
