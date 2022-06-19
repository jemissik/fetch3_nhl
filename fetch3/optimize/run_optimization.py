"""
FETCH3 optimization
-------------------
Runs optimization for FETCH3.
Options for the optimization are set in the .yml file.

This script is meant to be run from the command line, and the optimization configuration .yml
file is specified as a command line argument, for example::
      python run_optimization.py --config_path /Users/jmissik/Desktop/repos/fetch3_nhl/optimize/umbs_optimization_config.yml

See optimization_config_template.yml for an example configuration file.

See ``optimization_results.ipynb`` for an example of how to explore the optimization results.
"""

import datetime as dt
import logging
import shutil
import time
from pathlib import Path


import click
from boa import (
    WrappedJobRunner,
    get_experiment,
    get_scheduler,
    load_experiment,
    load_experiment_config,
    make_experiment_dir,
    save_experiment,
    scheduler_to_json_file,
    normalize_config
)

from fetch3.optimize.fetch_wrapper import Fetch3Wrapper

# from ax.storage.json_store.save import save_experiment
# from ax.storage.registry_bundle import RegistryBundle


@click.command()
@click.option(
    "-f",
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).resolve().parent.parent.parent / "config_files/opt_model_config.yml",
    help="Path to configuration YAML file.",
)
def main(config_file):
    """This is my docstring

    Args:
        config (os.PathLike): Path to configuration YAML file
    """
    run(config_file)


def run(config_file):
    start = time.time()

    wrapper = Fetch3Wrapper()
    config = wrapper.load_config(config_file)
    config = normalize_config(config)

    # config = load_experiment_config(config_file)  # Read experiment config'
    experiment_dir = make_experiment_dir(
        config["optimization_options"]["output_dir"],
        config["optimization_options"]["experiment_name"],
    )
    wrapper.experiment_dir = experiment_dir

    # Copy the experiment config to the experiment directory
    shutil.copyfile(config_file, experiment_dir / Path(config_file).name)

    log_format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(
        filename=Path(experiment_dir) / "optimization.log",
        filemode="w",
        format=log_format,
        level=logging.DEBUG,
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__file__)

    logger.info("Start time: %s", dt.datetime.now().strftime("%Y%m%dT%H%M%S"))

    # wrapper = Fetch3Wrapper(
    #     ex_settings=config["optimization_options"],
    #     model_settings=config["model_options"],
    #     experiment_dir=experiment_dir,
    # )

    experiment = get_experiment(config, WrappedJobRunner(wrapper=wrapper), wrapper)

    scheduler = get_scheduler(experiment, config=config)

    scheduler.run_all_trials()

    # save_experiment(experiment, "experiment.json")
    # print(load_experiment("experiment.json.pickle"))

    # metric = get_metric_from_config(config["optimization_options"]["metric"])
    # bundle = RegistryBundle(
    #     metric_clss={metric: None},
    #     runner_clss={WrappedJobRunner: None}
    # )
    # save_experiment(experiment, "experiment.json")
    # from ax.storage.json_store.load import load_experiment
    logging.info("\nTrials completed! Total run time: %d", time.time() - start)
    scheduler_to_json_file(scheduler, experiment_dir / "scheduler.json")
    return scheduler


if __name__ == "__main__":
    main()
