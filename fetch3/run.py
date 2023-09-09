# -*- coding: utf-8 -*-
"""
####
Main
####

Main model runner

Note: This is intended to be run from the command line
"""
import time
import os

start = time.time()  # start run clock

import shutil
import logging
import yaml
from pathlib import Path
import concurrent.futures

import click

from fetch3 import __version__ as VERSION
from fetch3.utils import make_experiment_directory
from fetch3.initial_conditions import initial_conditions
from fetch3.met_data import prepare_met_data
from fetch3.model_config import save_calculated_params, get_multi_config, ConfigParams
from fetch3.model_functions import Picard, format_model_output, save_nc, combine_outputs
from fetch3.model_setup import spatial_discretization, temporal_discretization
from fetch3.sapflux import calc_sapflux, format_inputs

log_format = "%(levelname)s %(asctime)s %(processName)s - %(name)s - %(message)s"

logging.basicConfig(
    filemode="w",
    format=log_format,
    level=logging.DEBUG
)
logger = logging.getLogger("fetch3")

logger.addHandler(logging.StreamHandler())

parent_path = Path(__file__).resolve().parent.parent
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
def main(config_path, data_path, output_path, species):
    cfgs = get_multi_config(config_path=config_path, species=species)

    model_options = cfgs[0].model_options

    # If using the default output directory, create directory if it doesn't exist
    if output_path == parent_path:
        output_path = default_output_path
        output_path.mkdir(exist_ok=True)

    # Make a new experiment directory if make_experiment_dir=True was specified in the config
    # Otherwise, use the output directory for the experiment directory
    mk_exp_dir = model_options.make_experiment_dir
    exp_name = model_options.experiment_name
    if mk_exp_dir:
        exp_dir = make_experiment_directory(output_path, experiment_name=exp_name)
    else:
        exp_dir = output_path

    log_path = exp_dir / "fetch3.log"
    if log_path.exists():
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    HEADER_BAR = """
    ##############################################
    """
    LOG_INFO = (
        f"""
    FETCH Run
    Output Experiment Dir: {exp_dir}
    Config file: {config_path}
    Start Time: {time.ctime(start)}
    Using config file: {config_path}
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


    try:
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            species_runs = {executor.submit(run_single, cfg, data_path, exp_dir): cfg for cfg in cfgs}
            logger.info("submitted jobs!")
            for future in concurrent.futures.as_completed(species_runs):
                original_task = species_runs[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.exception('%r generated an exception: %s' % (original_task, exc))
            concurrent.futures.wait(species_runs)
        nc_output = combine_outputs(results)
        save_nc(exp_dir, nc_output)
    except Exception as e:
        logger.exception("Error completing Run! Reason: %r", e)
        raise
    finally:
        logger.info(f"run time: {time.time() - start} s")  # end run clock
        logger.info("run complete")


def run_single(cfg: ConfigParams, data_dir, output_dir):
    log_path = output_dir / f"fetch3_{cfg.species}.log"
    if log_path.exists():
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    # Log the directories being used
    logger.info("Using output directory: " + str(output_dir))

    # save the calculated params to a file
    save_calculated_params(str(output_dir / "calculated_params.yml"), cfg)

    ##########Set up spatial discretization
    zind = spatial_discretization(cfg)
    ######prepare met data
    met, tmax, start_time, end_time = prepare_met_data(cfg, data_dir, zind.z_upper)

    t_num, nt = temporal_discretization(cfg, tmax)
    logger.info("Total timesteps to calculate: : %d" % nt)

    ############## Calculate initial conditions #######################
    logger.info("Calculating initial conditions ")
    H_initial, Head_bottom_H = initial_conditions(cfg, met.q_rain, zind)

    ############## Run the model #######################
    logger.info("Running the model ")
    (
        H,
        K,
        S_stomata,
        theta,
        S_kx,
        S_kr,
        C,
        Kr_sink,
        Capac,
        S_sink,
        EVsink_ts,
        THETA,
        infiltration,
        trans_2d,
        nhl_trans_2d,
    ) = Picard(cfg, H_initial, Head_bottom_H, zind, met, t_num, nt, output_dir, data_dir)

    ############## Calculate water balance and format model outputs #######################
    df_waterbal, df_EP, nc_output = format_model_output(
        cfg.species,
        H,
        K,
        S_stomata,
        theta,
        S_kx,
        S_kr,
        C,
        Kr_sink,
        Capac,
        S_sink,
        EVsink_ts,
        THETA,
        infiltration,
        trans_2d,
        nhl_trans_2d,
        cfg.model_options.dt,
        start_time,
        end_time,
        cfg.model_options.dz,
        cfg,
        zind,
    )

    # Calculate sapflux and aboveground storage
    ds_sapflux = calc_sapflux(nc_output["ds_canopy"], cfg)

    nc_output["sapflux"] = ds_sapflux

    logger.info("Finished running species: %s", cfg.species)

    return nc_output


if __name__ == "__main__":
    main()
