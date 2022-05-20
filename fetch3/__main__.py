# -*- coding: utf-8 -*-
"""
####
Main
####

Main model runner

Note: This is intended to be run from the command line
"""
import time
start = time.time()  # start run clock

import click
import logging
from pathlib import Path

from fetch3.model_setup import spatial_discretization, temporal_discretization
from fetch3.model_config import setup_config, save_calculated_params
from fetch3.met_data import prepare_met_data



from fetch3.initial_conditions import initial_conditions
from fetch3.model_functions import format_model_output, Picard, save_csv, save_nc

from fetch3.sapflux import calc_sapflux, format_inputs

logger = logging.getLogger(__file__)

parent_path = Path(__file__).resolve().parent.parent
default_config_path = parent_path / 'config_files' / 'model_config.yml'
default_data_path = parent_path / 'data'
default_output_path = parent_path / 'output'
model_dir = Path(__file__).parent.resolve() # File path of model source code


@click.command()
@click.option("--config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=str(default_config_path),
              help="Path to configuration YAML file")
@click.option("--data_path", type=click.Path(exists=True, path_type=Path),
              default=str(default_data_path),
              help="Path to data directory")
@click.option("--output_path", type=click.Path(exists=True, path_type=Path),
              default=str(default_output_path),
              help="Path to output directory")
def main(config_path, data_path, output_path):
    run(config_path, data_path, output_path)


def run(config_file, data_dir, output_dir):
    # If using the default output directory, create directory if it doesn't exist
    if output_dir == default_output_path:
        (output_dir).mkdir(exist_ok=True)

    # Start logger
    start_logger(output_dir=output_dir)
    logger = logging.getLogger(__file__)
    try:
        # Log the directories being used
        logger.info("Using config file: " + str(config_file) )
        logger.info("Using output directory: " + str(output_dir) )

        cfg = setup_config(config_file)

        #save the calculated params to a file
        save_calculated_params(str(output_dir / 'calculated_params.yml'), cfg)

        ##########Set up spatial discretization
        zind = spatial_discretization(
        cfg.dz, cfg.Soil_depth, cfg.Root_depth, cfg.Hspec, cfg.sand_d, cfg.clay_d)
        ######prepare met data
        met, tmax, start_time, end_time = prepare_met_data(cfg, data_dir, zind.z_upper)

        t_num, nt = temporal_discretization(cfg, tmax)
        logger.info("Total timesteps to calculate: : %d" % nt)

        ############## Calculate initial conditions #######################
        logger.info("Calculating initial conditions " )
        H_initial, Head_bottom_H = initial_conditions(cfg, met.q_rain, zind)


        ############## Run the model #######################
        logger.info("Running the model ")
        H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts, THETA, infiltration,trans_2d = Picard(cfg, H_initial, Head_bottom_H, zind, met, t_num, nt, output_dir, data_dir)

        ############## Calculate water balance and format model outputs #######################
        df_waterbal, df_EP, nc_output = format_model_output(H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink, EVsink_ts, THETA,
                        infiltration,trans_2d, cfg.dt, start_time, end_time, cfg.dz, cfg, zind)

        # Calculate sapflux and aboveground storage
        H_above, trans_2d_tree = format_inputs(nc_output['ds_canopy'], cfg.mean_crown_area_sp)

        ds_sapflux = calc_sapflux(H_above, trans_2d_tree, cfg)

        nc_output['sapflux'] = ds_sapflux

        ####################### Save model outputs ###################################
        logger.info("Saving outputs")
        save_csv(output_dir, df_waterbal, df_EP)
        save_nc(output_dir, nc_output)
    except Exception as e:
        logger.exception("Error completing Run! Reason: %s" % e)

    finally:
        logger.info(f"run time: {time.time() - start} s")  # end run clock
        logger.info("run complete")



def start_logger(output_dir):
    log_format = "%(levelname)s %(asctime)s - %(message)s"

    logging.basicConfig(filename=output_dir / "fetch3.log",
                        filemode="w",
                        format=log_format,
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == "__main__":
    main()