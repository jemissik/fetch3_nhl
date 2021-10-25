from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


files = [
    "C.csv",
    "Capac.csv",
    "df_EP.csv",
    "df_waterbal.csv",
    "EVsink_total.csv",
    "EVsink_ts.csv",
    "H.csv",
    "infiltration.csv",
    "K.csv",
    "Kr_sink.csv",
    "S_kr.csv",
    "S_kx.csv",
    "S_stomata.csv",
    "theta.csv",
    "trans_2d.csv",
]

output_dir = Path("output")
data_dir = Path("tests/data")

def test_output_data_should_be_the_same_as_previously_stored_data():
    # Given previous ran data

    # When we run the model
    from main import Picard

    # Then the output data and the previous data should be the same
    for file in files:
        previous_df = pd.read_csv(data_dir / file)
        output_df = pd.read_csv(output_dir / file)
        assert_frame_equal(previous_df, output_df), file