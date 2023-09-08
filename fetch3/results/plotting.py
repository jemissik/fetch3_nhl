import hvplot.pandas
import hvplot.xarray


def plot_2d_slice(ds=None, z=None, var=None):

    plot = ds[var].sel(z=z).hvplot.line(x='time')
    return plot

def plot_precip_and_swc(model_ds, z=None, obs=None, obs_P='P_F', obs_swc='SWC_F_MDS_1', scale_obs_swc=True):

    obs_swc = obs[obs_swc]

    # If obs are in % instead of fraction
    if scale_obs_swc:
        obs_swc = obs_swc / 100

    precip_plot = obs[obs_P].hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_obs_plot = obs_swc.hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_model_plot = model_ds.THETA.sel(z=z).hvplot.line(x='time', label='model SWC')
    swc_plot = (swc_obs_plot * swc_model_plot).opts(ylim=(0,0.3))
    return (precip_plot + swc_plot).cols(1)