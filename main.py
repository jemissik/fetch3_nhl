# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:10:31 2019

@author: mdef0001
"""
import time
start = time.time()  # start run clock

from initial_conditions import initial_conditions
from model_functions import format_model_output, Picard, save_csv, save_nc
import model_config as cfg

############## Calculate initial conditions #######################
H_initial, Head_bottom_H = initial_conditions()

############## Run the model #######################
H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts, THETA, infiltration,trans_2d = Picard(H_initial, Head_bottom_H)

############## Calculate water balance and format model outputs #######################
df_waterbal, df_EP, nc_output = format_model_output(H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink, EVsink_ts,
                                                     THETA, infiltration,trans_2d, cfg.dt, cfg.dz)

####################### Save model outputs ###################################
save_csv(df_waterbal, df_EP)
save_nc(nc_output)

print(f"run time: {time.time() - start} s")  # end run clock