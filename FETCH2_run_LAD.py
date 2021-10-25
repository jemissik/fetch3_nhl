# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:10:31 2019

@author: mdef0001
"""
import time
start = time.time() #start run clock

from model_functions import *

############## inital condition #######################

#setting profile for initial condition
if BottomBC==0:
    H_initial[0]=Head_bottom_H[0]


# if __name__ == '__main__':
H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts, THETA, infiltration,trans_2d = Picard(H_initial)  #calling the function



######################################################################################

####################### Water balance ###################################

theta_i=sum(THETA[:,1]*dz)
theta_t=sum(THETA[:,-1]*dz)
theta_tot=theta_i-theta_t  #(m)
theta_tot=theta_tot*1000  #(mm)

infilt_tot=sum(infiltration)*dt*1000 #mm
if UpperBC==0:
    theta_tot=(theta_tot)+infilt_tot
############################

EVsink_total=np.zeros(shape=(len(EVsink_ts[0])))
for i in np.arange(1,len(EVsink_ts[0]),1):
    EVsink_total[i]=sum(-EVsink_ts[:,i]*dz)  #(1/s) over the simulation times dz [m]= m

root_water=sum(EVsink_total)*1000*dt #mm
#############################

transpiration_tot=sum(sum(trans_2d))*1000*dt*dz ##mm

df_waterbal = pd.DataFrame(data={'theta_i':theta_i,
            'theta_t':theta_t, 'theta_tot':theta_tot, 'infilt_tot':infilt_tot,
                'root_water':root_water, 'transpiration_tot':transpiration_tot}, index = [0])

#summing during all time steps and multiplying by 1000 = mm  #
#the dt factor is accounting for the time step - to the TOTAl and not the rate

#end of simulation adding +1 time step to match dimensions
step_time = pd.Series(pd.date_range(start_time, end_time + pd.to_timedelta(dt, unit = 's'), freq=str(dt)+'s'))
############################################################################

df_time = pd.DataFrame(data=step_time.index.values,index=step_time)

#########################################################


d = {'trans':(sum(trans_2d[:,:]*dz)*1000)} #mm/s
df_EP = pd.DataFrame(data=d,index=step_time[:])

trans_h=dt*df_EP['trans'].resample('60T').sum() # hourly accumulated simulated transpiration


#output
output_vars = {'H':H, 'K': K, 'S_stomata':S_stomata, 'theta':theta, 'S_kx':S_kx, 'S_kr':S_kr, 'C':C,
            'Kr_sink':Kr_sink, 'Capac':Capac, 'EVsink_ts':EVsink_ts,
            'THETA':THETA, 'infiltration':infiltration, 'trans_2d':trans_2d,'EVsink_total':EVsink_total}

for var in output_vars:
    pd.DataFrame(output_vars[var]).to_csv(working_dir / 'output' / (var + '.csv'), index = False, header=False)

df_waterbal.to_csv(working_dir / 'output' / ('df_waterbal' + '.csv'), index=False, header=True)
df_EP.to_csv(working_dir / 'output' / ('df_EP' + '.csv'), index=False, header=True)


print(f"run time: {time.time() - start} s")