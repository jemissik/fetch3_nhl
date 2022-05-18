###############
Quick reference
###############

****************
Working with git
****************

To merge upstream changes into your branch::
    
    git fetch upstream
    git merge upstream/develop

******************
Working with conda
******************

To activate the fetch3 environment::

    conda activate fetch3-dev

**************
Running FETCH3
**************

Running FETCH3 (without optimization)::

    python main.py --config_path <path to your config file>
    --data_path <path to your data folder> --output_path <path to your output directory>

Running an optimization::

    python optimization_run.py  --config_file <your_config_path>

.. note::
    Make sure that you have the fetch3-dev environment activated and you are inside the fetch3_nhl
    directory when you run the model.