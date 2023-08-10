"""
This file runs NHL as a standalone module (without running FETCH3).

It returns NHL transpiration, and also writes NHL
transpiration to a netcdf file.

"""
import time
import os

start = time.time()  # start run clock
import shutil
import logging

import fetch3.nhl_transpiration.main as nhl
import click
from pathlib import Path
from fetch3.model_config import save_calculated_params, get_single_config
from fetch3.utils import make_experiment_directory
from fetch3 import __version__ as VERSION

log_format = "%(levelname)s %(asctime)s %(processName)s - %(name)s - %(message)s"

# logging.basicConfig(
#     filemode="w",
#     format=log_format,
#     level=logging.DEBUG
# )

DEFAULT_LEVEL = logging.DEBUG

logger = logging.getLogger("fetch3.nhl_transpiration.main_standalone")
logger.setLevel(DEFAULT_LEVEL)
sh = logging.StreamHandler()
sh.setLevel(DEFAULT_LEVEL)
sh.setFormatter(logging.Formatter(log_format))
logger.addHandler(sh)

parent_path = Path(__file__).resolve().parent.parent.parent
default_config_path = parent_path / "config_files" / "model_config.yml"
default_data_path = parent_path / "data"
default_output_path = parent_path / "output"
model_dir = Path(__file__).parent.resolve()  # File path of model source code

@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(default_config_path),
    help="Path to configuration YAML file",
)
@click.option(
    "--data_path",
    type=click.Path(exists=True, path_type=Path),
    default=str(default_data_path),
    help="Path to data directory",
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, path_type=Path),
    default=str(parent_path),
    help="Path to output directory",
)
@click.option(
    "--species",
    type=str,
    default=None,
    help="species to run the model for"
)
def run(config_path, data_path, output_path, species):
    cfg = get_single_config(config_path=config_path, species=species)

    # If using the default output directory, create directory if it doesn't exist
    if output_path == parent_path:
        output_path = default_output_path
        output_path.mkdir(exist_ok=True)

    # Make a new experiment directory if make_experiment_dir=True was specified in the config
    # Otherwise, use the output directory for the experiment directory
    mk_exp_dir = cfg.model_options.make_experiment_dir
    exp_name = cfg.model_options.experiment_name
    if mk_exp_dir:
        exp_dir = make_experiment_directory(output_path, experiment_name=exp_name)
    else:
        exp_dir = output_path

    log_path = exp_dir / "nhl.log"
    if log_path.exists():
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    fh.setLevel(DEFAULT_LEVEL)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    HEADER_BAR = """
    ##############################################
    """
    LOG_INFO = (
        f"""
    NHL Standalone Run
    Output Experiment Dir: {exp_dir}
    Config file: {config_path}
    Start Time: {time.ctime(start)}
    Version: {VERSION}"""
    )

    logger.info(
        f"\n{HEADER_BAR}"
        f"\n{LOG_INFO}"
        f"\n{HEADER_BAR}"
    )
    # Copy the config file to the output directory
    copied_config_path = exp_dir / config_path.name
    if not copied_config_path.exists():
        shutil.copy(config_path, copied_config_path)

    # save the calculated params to a file
    save_calculated_params(str(exp_dir / "calculated_params.yml"), cfg)


    try:
        logger.info("Running NHL transpiration...")

        # NHL in units of [kg H2O m-2crown_projection s-1 m-1stem]
        nhl_trans_tot, LAD = nhl.main(cfg, exp_dir, data_path, to_model_res=False, write_output=True)
    except Exception as e:
        logger.exception("Error completing Run! Reason: %r", e)
        raise
    finally:
        logger.info(f"run time: {time.time() - start} s")  # end run clock
        logger.info("run complete")


if __name__ == "__main__":
    run()