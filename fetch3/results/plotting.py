import hvplot.pandas
import hvplot.xarray

import holoviews as hv
from holoviews import dim, opts
from fetch3.results.results import Results


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


def plot_precip_swc(res, swc_var='SWC', p_var='P_F', swc_lim=(0, 50), p_lim=(0, 30), swc_label="SWC (%)", p_label="P (mm)", **kwargs):
    df=res.obs
    swc_plot = hv.Curve(df, kdims=df.index.name, vdims=swc_var).opts(color="k", ylim=swc_lim, ylabel=swc_label)
    p_plot = hv.Curve(df, kdims=df.index.name, vdims=p_var).opts(ylim=p_lim, invert_yaxis=True, color="#30a2da", ylabel=p_label)
    overlay =  swc_plot * p_plot
    return overlay.opts(multi_y=True, xlim=(res.canopy1d.time.min().values, res.canopy1d.time.max().values), **kwargs).redim(TIMESTAMP='time')


def plot_precip_swc_vpd(res, swc_var='SWC', p_var='P_F', irr_var=None, vpd_var='VPD_F', swc_lim=(0, None), p_lim=(0, None), vpd_lim=(0,None), swc_label="SWC (%)", p_label="P (mm)", p_ax=None, **kwargs):

    if isinstance(res, Results):
        df=res.obs
        tmin = res.canopy1d.time.min().values
        tmax = res.canopy1d.time.max().values
    else:
        df=res
        tmin = df.index.min()
        tmax = df.index.max()

    rename = {df.index.name: 'time'}

    p_plot = hv.Curve(df, kdims=df.index.name, vdims=p_var, label="P (mm)").opts(ylim=p_lim, invert_yaxis=True, color="#47a5e3", ylabel=p_label, yaxis=p_ax, **kwargs)
    swc_plot = hv.Curve(df, kdims=df.index.name, vdims=swc_var, label="SWC").opts(color="k", ylim=swc_lim, ylabel=swc_label, **kwargs)

    if irr_var is not None:
        irr_plot = hv.Curve(df, kdims=df.index.name, vdims=irr_var, label="Irrigation (mm)").opts(ylim=p_lim, invert_yaxis=True, color="#0b5ca3", yaxis=None, **kwargs)

    vpd_plot = hv.Curve(df, kdims=df.index.name, vdims=vpd_var, label="VPD").opts(ylim=vpd_lim, ylabel="VPD (hPa)", color='#de8f06', **kwargs)

    if irr_var is not None:
        overlay =  swc_plot * p_plot * irr_plot * vpd_plot
    else:
        overlay =  swc_plot * p_plot * vpd_plot
    return overlay.opts(multi_y=True, xlim=(tmin, tmax)).redim(**rename)


def plot_sap(res, obs_df, tree_name=None, **kwargs):
    canopy1d = res.canopy1d
    df = res.obs[obs_df]
    if tree_name is None:
        tree_name = res.cfg.species
    rename = {df.index.name: 'time'}
    mod_sap = hv.Curve(canopy1d, kdims='time', vdims='trans', label="Model sapflow").opts(**kwargs)
    nhl = hv.Curve(canopy1d, kdims='time', vdims='nhl', label="Potential Transp.").opts(**kwargs)
    obs_sap = hv.Curve(df, kdims='TIMESTAMP', vdims=tree_name, label="Observed sapflow").redim(**rename).opts(**kwargs)

    return (nhl * mod_sap * obs_sap).opts(ylabel="Sapflow (cm3 hr-1)", responsive=True,)
