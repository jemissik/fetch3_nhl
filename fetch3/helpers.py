"""
Convenience functions:
- loading model outputs
"""
import xarray as xr
from pathlib import Path

def load_model_outputs(model_output_path):

    filein = Path(model_output_path) / "ds_all.nc"
    dsall = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_canopy.nc"
    canopy = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_soil.nc"
    soil = xr.load_dataset(filein)

    filein = Path(model_output_path) / "ds_root.nc"
    roots = xr.load_dataset(filein)

    filein = Path(model_output_path) / "sapflux.nc"
    sapflux = xr.load_dataset(filein)

    filein = Path(model_output_path) / "nhl_modelres_trans_out.nc"
    nhl = xr.load_dataset(filein)

    return dsall, canopy, soil, roots, sapflux, nhl