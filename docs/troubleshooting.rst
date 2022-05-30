###############
Troubleshooting
###############

*************
Common issues
*************

``ModuleNotFoundError``
=======================

* Are you sure you're using the fetch3 conda environment? (Remember that if you
  open a new terminal, you need to activate the environment again)
* Depending on the terminal you're using, you should probably see ``fetch3-dev`` in parentheses somewhere in the command line.
  If you see ``base`` in parentheses, this means you're using the base conda environment instead of the fetch3 environment.

  * You can run ``conda env list`` to check which environment is active (there is a ``*`` next to the active environment)
  * If you're running one of the .ipynb notebooks to look at the model output, make sure that the notebook is using the fetch3-dev environment

* If you're sure you're using the fetch3-dev environment and still getting a ``ModuleNotFoundError``, see :ref:`Installation errors`

``No such file or directory``
=============================

* Are you sure you're using the correct directory?

  * For example, if you get this error when you run ``python main.py``, run ``ls`` and check that there is a ``main.py`` file
    in the current directory.
  * If you change the directories in the notebooks, double-check that you're giving it the correct directory.
  * If you changed the output path for the model, make sure you specified an output directory that exists on your computer.
  * Check whether a full filepath or a directory was supposed to be provided, and that you provided the correct one
    (e.g., if the error was with the path to the config file, did you provide the full filepath including the
    filename and extension, or just a directory?)

* For directory errors in Windows:

  * If you're sure  you're using the correct directory and still getting an error, it's probably an issue with parsing
    the Windows path correctly. There might be characters (like spaces) that need to be escaped for the path to be
    parsed properly.

    * Try using double backslashes in the file paths (i.e., change ``\`` to ``\\``)
    * Try keeping things in directories that don't have any special characters or
      spaces in the path


Installation errors
===================

* For Windows, make sure git is installed before you install the fetch3 conda environment
* If you are using the fetch3 environment and are still getting a ``ModuleNotFoundError``, check that the module in question was actually installed in
  the fetch3-dev environment. With the fetch3-dev environment active, run ``conda list`` to see the packages that are installed in fetch3-dev.
* If you find that the fetch3 environment is missing packages that should have been installed, try removing and re-installing the
  environment.

  * To remove the fetch3 environment::

        conda deactivate
        conda remove -n fetch3-dev --all

  * Next, make sure you have the latest version of the FETCH3 code, and re-install the fetch3-dev environment. See :ref:`Install FETCH3’s dependencies`

Permissions issues in Windows
=============================

* If you're getting an error about not having permission to access files in Windows, you might need to change
  your permissions settings.

.. todo::
    Add instructions about changing permissions settings in powershell to run as admin
    by default


************
Other issues
************

If you're having an issue not covered in the above section, make sure you've updated everything to the latest version.

- Make sure you have the latest version of the FETCH3 code from GitHub
- Using the most recent version of the FETCH3 repository, update your conda environment. With the fetch3-dev environment
  active, run::

    conda env update -file fetch3_requirements.yml --prune

- Update boa::

    pip install -U git+https://github.com/madeline-scyphers/boa.git


If you're still having issues, please `submit a GitHub issue <https://github.com/jemissik/fetch3_nhl/issues>`_
