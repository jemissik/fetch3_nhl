from pathlib import Path
import scipy.io
import numpy as np

import pandas as pd
from pandas.testing import assert_frame_equal



files = [
    "gb",
    "geff",
    "gs",
    "P0",
    "Qp",
    "A",
    "Km",
    "U",
    "Ci",
    "LAD",
    "NHL_trans_leaf",
    "NHL_trans_sp_crownarea"

]

data_dir = Path("tests/data")
output_dir = Path("output")


def test_output_data_should_be_the_same_as_previously_stored_data():
    # Given previous ran data

    # When we run the model
    from main import NHL_trans_sp_stem

    # Then the output data and the previous data should be the same
    for file in files:
        mat = scipy.io.loadmat(str(data_dir / (file + '.mat')), squeeze_me =True)
        mat = {list(mat)[-1]: mat[list(mat)[-1]]}
        if type(mat[list(mat)[-1]]) == float:
            mat = pd.DataFrame(mat, index = [0])
        else:
            mat = pd.DataFrame(mat)
        mat.columns = [0]

        output_df = pd.read_csv(output_dir / (file + '.csv'), header=None)

        # output_df = output_df.drop(output_df.index[-1], axis = 0) #drop last timestep
        assert_frame_equal(mat, output_df, check_exact=False, rtol = 1e-3), file  # Selects length of saved data to match length of run