###############
Quick reference
###############

.. note::
    This page is intended to serve as a convenient reference of commonly-used commands for those who are already familiar
    with using the model. Please read the full instruction pages first!

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


****************
Working with git
****************

To see the current state of your git repository::

    git status

To stage changes to be committed::

    git add <filename>  # Stages a specific file
    git add .  # Stages all changed files

To commit changes::

    git commit -m "<commit_message>"

To push changes to GitHub::

    git push

To merge upstream changes into your branch (run from the branch you want to merge changes into)::

    git fetch upstream
    git merge upstream/develop

To switch to another branch::

    git checkout <branch_name>

To create a new branch::

    git checkout -b <new_branch_name>

To stash changes (when you have changes you aren't ready to commit, but want to switch branches)::

    git stash

To apply changes that were stashed::

    git stash pop


******************
Working with conda
******************

To activate the fetch3 environment::

    conda activate fetch3-dev
