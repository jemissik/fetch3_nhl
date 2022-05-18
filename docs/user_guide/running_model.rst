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