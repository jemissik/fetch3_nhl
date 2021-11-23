from pathlib import Path

import pandas as pd
import xarray as xr
from pandas.testing import assert_frame_equal

varlist = ['Qp',
           'An',
           'gs',
           'Ci',
           'gb',
           'geff',
           'NHL_trans_leaf']

data_dir = Path("tests/data")
output_dir = Path("output")

#test for zenith angle

def test_output_data_should_be_the_same_as_previously_stored_data():
    # Given previous ran data

    # When we run the model
    from main import ds

    # Then the output data and the previous data should be the same

    #zenith angle
    test = pd.read_csv(data_dir / 'zenith.csv')
    out = pd.read_csv(output_dir / 'zenith.csv')
    assert_frame_equal(test, out, check_exact=False)

    #nc output
    test = xr.open_dataset(data_dir / 'out.nc')
    out = xr.open_dataset(output_dir / 'out.nc')

    for var in varlist:
        df1 = pd.DataFrame(test[var].values)
        df2 = pd.DataFrame(out[var].values)

        # output_df = output_df.drop(output_df.index[-1], axis = 0) #drop last timestep
        assert_frame_equal(df1, df2, check_exact=False), var  