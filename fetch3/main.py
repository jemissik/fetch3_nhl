# -*- coding: utf-8 -*-
"""
####
Main
####

Main model runner

Note: This is intended to be run from the command line
"""

import logging
from fetch3.model_config import cfg, output_dir

import time
start = time.time()  # start run clock

from fetch3.initial_conditions import initial_conditions
from fetch3.model_functions import format_model_output, Picard, save_csv, save_nc

from sapflux import calc_sapflux, format_inputs

logger = logging.getLogger(__file__)


def main():
    ############## Calculate initial conditions #######################
    logger.info("Calculating initial conditions " )
    H_initial, Head_bottom_H = initial_conditions()

    ############## Run the model #######################
    logger.info("Running the model ")
    H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts, THETA, infiltration,trans_2d = Picard(H_initial, Head_bottom_H)

    ############## Calculate water balance and format model outputs #######################
    df_waterbal, df_EP, nc_output = format_model_output(H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink, EVsink_ts,
                                                        THETA, infiltration,trans_2d, cfg.dt, cfg.dz)

    # Calculate sapflux and aboveground storage
    H_above, trans_kg = format_inputs(nc_output)

    ds_sapflux = calc_sapflux(H_above, trans_kg, cfg)

    nc_output['sapflux'] = ds_sapflux

    ####################### Save model outputs ###################################
    logger.info("Saving outputs")
    save_csv(output_dir, df_waterbal, df_EP)
    save_nc(output_dir, nc_output)

    logger.info(f"run time: {time.time() - start} s")  # end run clock


if __name__ == "__main__":
    main()