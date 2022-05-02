import pytest
from pathlib import Path

def test_fetch3_nhl_run():
    config_path = Path(__file__).resolve().parent.parent.parent / 'config_files' / 'test_nhl_oak.yml'
    data_path = Path(__file__).resolve().parent.parent.parent / 'data'
    output_path = Path(__file__).resolve().parent.parent.parent / 'output'
    from fetch3.__main__ import run
    run(config_path, data_path, output_path)

# /Users/jmissik/Desktop/repos/fetch3_nhl/output

# /Users/jmissik/Desktop/repos/fetch3_nhl/output