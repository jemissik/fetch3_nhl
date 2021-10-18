 ##PLOTS

plt.subplot(3, 1, 1)
#bottom and top of the canopy
plt.plot(step_time[:],H[nz_r,:],linewidth=0.8)
plt.plot(step_time[:],H[nz-1,:],linewidth=0.8)
plt.legend(['bottom of the canopy','top of the canopy'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - canopy')


plt.subplot(3, 1, 2)
#bottom and top of the roots
plt.plot(step_time[:],H[nz_s,:],linewidth=0.8)
plt.plot(step_time[:],H[nz_r-1,:],linewidth=0.8)
plt.legend(['bottom of the roots','top of the roots'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - roots')


plt.subplot(3, 1, 3)
#bottom and top of the soil - bottom with roots
plt.plot(step_time[:],H[nz_s-(nz_r-nz_s),:],linewidth=0.8)
plt.plot(step_time[:],H[nz_s-1,:],linewidth=0.8)
plt.legend([str(round(z[nz_s-(nz_r-nz_s)],2)),str(round(z[nz_s-1],2))], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - soil')


#bottom and top of the soil - bottom with roots
plt.plot(step_time[:],H[nz_s-1,:],linewidth=0.8)
plt.plot(step_time[:],H[0,:],linewidth=0.8)

plt.legend(['top of the soil'], loc=1, fontsize = 'small')
plt.ylabel('$\Phi$ (MPa) - soil')
