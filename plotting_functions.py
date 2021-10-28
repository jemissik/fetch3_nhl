import matplotlib.pyplot as plt
import pandas as pd
from FETCH2_loading_LAD import nz_s, nz_r,z, nz
from met_data import start_time, end_time
import model_config as cfg

fdata = 'output/H.csv'
H = pd.read_csv(fdata, header = None)

 ##PLOTS
step_time = pd.Series(pd.date_range(start_time, end_time + pd.to_timedelta(cfg.dt, unit = 's'), freq=str(cfg.dt)+'s'))


plt.subplot(3, 1, 1)
#bottom and top of the canopy
plt.plot(step_time[:],H.loc[:,nz_r],linewidth=0.8)
plt.plot(step_time[:],H.loc[:,nz-1],linewidth=0.8)
plt.legend(['bottom of the canopy','top of the canopy'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - canopy')


plt.subplot(3, 1, 2)
#bottom and top of the roots
plt.plot(step_time[:],H.loc[:,nz_s],linewidth=0.8)
plt.plot(step_time[:],H.loc[:,nz_r-1],linewidth=0.8)
plt.legend(['bottom of the roots','top of the roots'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - roots')


plt.subplot(3, 1, 3)
#bottom and top of the soil - bottom with roots
plt.plot(step_time[:],H.loc[:,nz_s-(nz_r-nz_s)],linewidth=0.8)
plt.plot(step_time[:],H.loc[:,nz_s-1],linewidth=0.8)
plt.legend([str(round(z[nz_s-(nz_r-nz_s)],2)),str(round(z[nz_s-1],2))], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - soil')


#bottom and top of the soil - bottom with roots
plt.plot(step_time[:],H.loc[:,nz_s-1],linewidth=0.8)
plt.plot(step_time[:],H.loc[:,0],linewidth=0.8)

plt.legend(['top of the soil'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - soil')
