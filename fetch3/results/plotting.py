import hvplot.pandas
import hvplot.xarray

import holoviews as hv
from holoviews import dim, opts
# from fetch3.results.results import Results

hv.plotting.bokeh.ElementPlot.width = 1200
hv.plotting.bokeh.ElementPlot.aspect = 4
hv.plotting.bokeh.ElementPlot.fontscale = 1.5

plot_kw = dict(
    responsive=True,
               tools=['hover'],
            aspect=4,
            active_tools=['box_zoom'],
            width=1200,
            fontscale=1.5,
              )

def xlabel_off(plot, element):
    plot.state.xaxis.major_label_text_alpha = 0


def plot_2d_slice(ds=None, z=None, var=None):

    plot = ds[var].sel(z=z).hvplot.line(x='time')
    return plot

def plot_precip_and_swc(model_ds, z=None, obs=None, obs_P='P_F', obs_swc='SWC_F_MDS_1', scale_obs_swc=True):

    # set z=0 if z is none

    obs_swc = obs[obs_swc]

    # If obs are in % instead of fraction
    if scale_obs_swc:
        obs_swc = obs_swc / 100

    precip_plot = obs[obs_P].hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_obs_plot = obs_swc.hvplot.line(xlim=(model_ds.time.min().values, model_ds.time.max().values))
    swc_model_plot = model_ds.THETA.sel(z=z, method='nearest').hvplot.line(x='time', label='model SWC')
    swc_plot = (swc_obs_plot * swc_model_plot).opts(ylim=(0,0.3))
    return (precip_plot + swc_plot).cols(1)


def plot_precip_swc(res, swc_var='SWC', p_var='P_F', swc_lim=(0, 50), p_lim=(0, 30), swc_label="SWC (%)", p_label="P (mm)", **kwargs):
    df=res.obs
    swc_plot = hv.Curve(df, kdims=df.index.name, vdims=swc_var).opts(color="k", ylim=swc_lim, ylabel=swc_label)
    p_plot = hv.Curve(df, kdims=df.index.name, vdims=p_var).opts(ylim=p_lim, invert_yaxis=True, color="#30a2da", ylabel=p_label)
    overlay =  swc_plot * p_plot
    return overlay.opts(multi_y=True, xlim=(res.canopy1d.time.min().values, res.canopy1d.time.max().values), **kwargs).redim(TIMESTAMP='time')


def plot_precip_swc_vpd(obs_df,
                        swc_var='SWC',
                        p_var='P_F',
                        irr_var=None,
                        vpd_var='VPD_F',
                        tlim=(None, None),
                        swc_lim=(0, None),
                        p_lim=(0, None),
                        vpd_lim=(0,None),
                        swc_label="SWC",
                        p_label="P (mm)",
                        scale_obs_swc=True,
                        p_ax=None,
                        show_xlabel=True,
                        **kwargs):

    df = obs_df.copy()

    # If obs are in % instead of fraction
    if scale_obs_swc:
        df[swc_var] = df[swc_var] / 100

    rename = {df.index.name: 'time'}
    df.index.name = 'time'

    p_plot = hv.Curve(df, kdims=df.index.name, vdims=p_var, label="P (mm)").opts(ylim=p_lim, invert_yaxis=True, color="#47a5e3", ylabel=p_label, yaxis=p_ax, **kwargs)
    swc_plot = hv.Curve(df, kdims=df.index.name, vdims=[swc_var], label="SWC").opts(ylim=swc_lim, ylabel=swc_label, color='k', **kwargs)

    if irr_var is not None:
        irr_plot = hv.Curve(df, kdims=df.index.name, vdims=irr_var, label="Irrigation (mm)").opts(ylim=p_lim, invert_yaxis=True, color="#0b5ca3", yaxis=None, **kwargs)

    vpd_plot = hv.Curve(df, kdims=df.index.name, vdims=vpd_var, label="VPD").opts(ylim=vpd_lim, ylabel="VPD (hPa)", color='#de8f06', **kwargs)

    if irr_var is not None:
        overlay =  swc_plot * p_plot * irr_plot * vpd_plot
    else:
        overlay =  swc_plot * p_plot * vpd_plot

    plot_opts = dict(multi_y=True, xlim=tlim)
    if not show_xlabel:
        plot_opts['hooks'] = [xlabel_off]
        plot_opts['xlabel'] = ''
    return overlay.opts(**plot_opts).redim(**rename)


def plot_sap(mod_ds, obs_df, obs_var=None, show_xlabel=True, **kwargs):
    df = obs_df.copy()
    df.index.name = 'time'
    mod_sap = hv.Curve(mod_ds, kdims='time', vdims='trans', label="Model sapflow").opts(**kwargs)
    nhl = hv.Curve(mod_ds, kdims='time', vdims='nhl', label="Potential Transp.").opts(**kwargs)
    obs_sap = hv.Curve(df, kdims='time', vdims=obs_var, label="Observed sapflow").opts(**kwargs)

    plot_opts={}
    if not show_xlabel:
        plot_opts['hooks'] = [xlabel_off]
        plot_opts['xlabel'] = ''

    return (nhl * mod_sap * obs_sap).opts(ylabel="Sapflow (cm3 hr-1)", **plot_opts, **plot_kw)

def plot_stemwp(ds, ds_daily, tlim=(None,None), **kwargs):
    stem_wp = hv.Curve(ds, kdims='time', vdims='H', label="Stem water potential").opts(**kwargs)
    stem_wp_daily = hv.Curve(ds_daily, kdims='time', vdims='H_min', label="Daily min stem water pot.").opts(**kwargs)

    return (stem_wp * stem_wp_daily).opts(ylabel="Stem water potential", xlim=tlim, **plot_kw)
