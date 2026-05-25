"""
Utilities for comparing FETCH model outputs with observations.

The functions in this module are intentionally usable outside the optimization
wrapper, but keep the same signatures as the legacy BOA-facing helpers.
"""

import numpy as np
import pandas as pd
import xarray as xr

from fetch3.scaling import convert_sapflux_m3s_to_mm30min, convert_trans_m3s_to_cm3hr


def load_obs_data(filein, timevar):
    """Load observation data with a timezone-naive datetime index."""
    obsdf = pd.read_csv(filein, index_col=[timevar], parse_dates=[timevar])
    if obsdf.index.tz is not None:
        obsdf.index = obsdf.index.tz_localize(None)
    return obsdf


def normalize_model_obs(model, obs):
    """Normalize model and observation series by their own mean and standard deviation."""
    model = (model - model.mean()) / model.std()
    obs = (obs - obs.mean()) / obs.std()
    return model, obs


def _drop_first_last_time_steps(obsdf, model):
    obsdf = obsdf.iloc[1:-1]
    model = model.isel(time=np.arange(1, len(model.time) - 1))
    return obsdf, model


def _select_species(modelds, species):
    if "species" in modelds.dims:
        modelds = modelds.sel(species=species)
    return modelds


def _select_z(modelds, z=None):
    if "z" not in modelds.dims:
        return modelds

    if z is None:
        z = modelds.z.values.max()
    return modelds.sel(z=z, method="nearest")


def _filter_hour_range(df, hour_range):
    if hour_range:
        return df[(df.index.hour >= hour_range[0]) & (df.index.hour <= hour_range[1])]
    return df


def _aggregate_series(series, aggregation):
    if aggregation == "mean":
        return series.mean()
    if aggregation == "min":
        return series.min()
    if aggregation == "max":
        return series.max()
    if aggregation == "sum":
        return series.sum()
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def get_model_plot_trans(modelfile, obs_file, obs_var, output_var, obs_tvar="TIMESTAMP_START", **kwargs):
    """Compare plot-level transpiration against observations in mm 30 min-1."""
    obsdf = load_obs_data(obs_file, obs_tvar)

    modelds = xr.load_dataset(modelfile)
    modelds = modelds.sel(species=output_var)

    obsdf = obsdf.loc[modelds.time.data[0] : modelds.time.data[-1]]
    modelds["sapflux_plot_mm30min"] = convert_sapflux_m3s_to_mm30min(modelds.sapflux_plot)

    obsdf, model = _drop_first_last_time_steps(obsdf, modelds.sapflux_plot_mm30min)

    not_nans = ~obsdf[obs_var].isna()
    obs_not_nans = obsdf[obs_var].loc[not_nans]
    model_not_nans = model.data[not_nans]

    return model_not_nans, obs_not_nans


def get_model_sapflux(
    modelfile,
    obs_file,
    obs_var,
    output_var,
    obs_tvar="TIMESTAMP",
    hour_range=None,
    normalize=True,
    **kwargs,
):
    """Compare tree-level model sapflux against sapflow observations in cm3 hr-1."""
    obsdf = load_obs_data(obs_file, obs_tvar)

    modelds = xr.load_dataset(modelfile)
    modelds = _select_species(modelds, output_var)

    modelds["sapflux_scaled"] = convert_trans_m3s_to_cm3hr(modelds.sapflux)
    modeldf = modelds.squeeze(drop=True).to_dataframe()

    df = pd.merge(modeldf, obsdf[[obs_var]], how="left", right_index=True, left_index=True, suffixes=["model", "obs"])
    df = df.dropna()

    if normalize:
        df["sapflux_scaled"], df[obs_var] = normalize_model_obs(df["sapflux_scaled"], df[obs_var])

    df = _filter_hour_range(df, hour_range)

    return df["sapflux_scaled"], df[obs_var]


def get_model_nhl_trans(
    modelfile,
    obs_file,
    obs_var,
    output_var,
    hour_range=None,
    normalize=False,
    use_daily=False,
    scaling_factor=None,
    obs_tvar="TIMESTAMP",
    **kwargs,
):
    """Compare standalone NHL tree transpiration against observations in cm3 hr-1."""
    obsdf = load_obs_data(obs_file, obs_tvar)

    modelds = xr.load_dataset(modelfile)
    modelds = _select_species(modelds, output_var)

    # NHL standalone output is kg H2O s-1; convert kg to m3 before cm3 hr-1.
    modelds["nhl_scaled"] = convert_trans_m3s_to_cm3hr(modelds.NHL_trans_sp_stem * 10**-3)
    modeldf = modelds.squeeze(drop=True).to_dataframe()

    df = pd.merge(modeldf, obsdf[[obs_var]], how="left", right_index=True, left_index=True, suffixes=["model", "obs"])
    df = df.iloc[1:-1]

    if use_daily:
        df = df.resample("D").agg(pd.Series.sum, skipna=False)

    df = df.dropna()

    if scaling_factor:
        df[obs_var] = df[obs_var] * scaling_factor

    if not use_daily:
        df = _filter_hour_range(df, hour_range)

    if normalize:
        df["nhl_scaled"], df[obs_var] = normalize_model_obs(df["nhl_scaled"], df[obs_var])

    return df["nhl_scaled"], df[obs_var]


def get_model_swc(
    modelfile,
    obs_file,
    obs_var,
    output_var,
    species,
    obs_tvar="TIMESTAMP_START",
    percent_units=True,
    obs_depth=0.1,
    **kwargs,
):
    """Compare soil water content at a depth below the soil surface."""
    modelds = xr.load_dataset(modelfile)
    z_soil_surface = modelds.z.max().values
    obs_z = z_soil_surface - obs_depth

    obs_multiplier = 0.01 if percent_units else None
    return get_model_obs(
        modelfile,
        obs_file=obs_file,
        obs_var=obs_var,
        output_var=output_var,
        species=species,
        obs_tvar=obs_tvar,
        obs_multiplier=obs_multiplier,
        obs_z=obs_z,
        **kwargs,
    )


def get_model_obs(
    modelfile,
    obs_file,
    obs_var,
    output_var,
    species,
    obs_tvar="TIMESTAMP_START",
    obs_multiplier=True,
    obs_z=None,
    normalize=False,
    **kwargs,
):
    """
    Read observation data and model output for a trial.

    This can be used for 1D and 2D model outputs where observations need only a
    scalar multiplier to convert to the same units as the model output. For 2D
    outputs, observations are compared with the nearest model z-slice.
    """
    obsdf = load_obs_data(obs_file, obs_tvar)

    if obs_multiplier:
        obsdf[obs_var] = obsdf[obs_var] * obs_multiplier

    modelds = xr.load_dataset(modelfile)
    modelds = _select_species(modelds, species)
    modelds = _select_z(modelds, obs_z)

    obsdf = obsdf.loc[modelds.time.data[0] : modelds.time.data[-1]]
    obsdf, model = _drop_first_last_time_steps(obsdf, modelds[output_var])

    not_nans = ~obsdf[obs_var].isna()
    obs_not_nans = obsdf[obs_var].loc[not_nans]
    model_not_nans = model.isel(time=not_nans.to_numpy()).data.transpose()

    if normalize:
        model_not_nans, obs_not_nans = normalize_model_obs(model_not_nans, obs_not_nans)

    return model_not_nans, obs_not_nans


def get_model_obs_summary(
    modelfile,
    obs_file,
    obs_var,
    output_var,
    species,
    obs_tvar="TIMESTAMP_START",
    obs_multiplier=True,
    obs_z=None,
    hour_range=None,
    resample_freq="D",
    model_aggregation="mean",
    obs_aggregation=None,
    normalize=False,
    **kwargs,
):
    """
    Compare aggregated model and observation summaries.

    This is useful when the optimization target is a daily or nighttime
    statistic rather than every half-hour observation, for example daily
    minimum canopy water potential or predawn mean soil water potential.
    """
    obsdf = load_obs_data(obs_file, obs_tvar)

    if obs_multiplier:
        obsdf[obs_var] = obsdf[obs_var] * obs_multiplier

    modelds = xr.load_dataset(modelfile)
    modelds = _select_species(modelds, species)
    modelds = _select_z(modelds, obs_z)

    obsdf = obsdf.loc[modelds.time.data[0] : modelds.time.data[-1]]
    obsdf, model = _drop_first_last_time_steps(obsdf, modelds[output_var])

    df = pd.DataFrame(
        {
            "model": np.asarray(model.data).squeeze(),
            "obs": obsdf[obs_var].to_numpy(),
        },
        index=obsdf.index,
    )
    df = _filter_hour_range(df, hour_range)

    obs_aggregation = obs_aggregation or model_aggregation
    model_summary = df["model"].resample(resample_freq).apply(_aggregate_series, aggregation=model_aggregation)
    obs_summary = df["obs"].resample(resample_freq).apply(_aggregate_series, aggregation=obs_aggregation)

    summary = pd.concat([model_summary.rename("model"), obs_summary.rename("obs")], axis=1).dropna()
    model_values = summary["model"]
    obs_values = summary["obs"]

    if normalize:
        model_values, obs_values = normalize_model_obs(model_values, obs_values)

    return model_values, obs_values
