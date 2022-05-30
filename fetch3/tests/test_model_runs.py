from pathlib import Path

import pytest


def test_fetch3_nhl_run():
    config_path = Path(__file__).resolve().parent.parent.parent / 'config_files' / 'model_config.yml'
    data_path = Path(__file__).resolve().parent.parent.parent / 'data'
    output_path = Path(__file__).resolve().parent.parent.parent / 'output'
    from fetch3.__main__ import run
    run(config_path, data_path, output_path)


def test_fetch3_nhl_run_optconfig():
    config_path = Path(__file__).resolve().parent.parent.parent / 'config_files' / 'opt_model_config.yml'
    data_path = Path(__file__).resolve().parent.parent.parent / 'data'
    output_path = Path(__file__).resolve().parent.parent.parent / 'output'
    print(output_path)
    from fetch3.__main__ import run
    run(config_path, data_path, output_path)

def test_fetch3_PM_run():
    config_path = Path(__file__).resolve().parent.parent.parent / 'config_files' / 'model_config_PM.yml'
    data_path = Path(__file__).resolve().parent.parent.parent / 'data'
    output_path = Path(__file__).resolve().parent.parent.parent / 'output'
    print(output_path)
    from fetch3.__main__ import run
    run(config_path, data_path, output_path)

def test_fetch3_opt_run():
    config_path = Path(__file__).resolve().parent.parent.parent / 'config_files' / 'opt_model_config.yml'
    output_path = Path(__file__).resolve().parent.parent.parent / 'output'
    print(output_path)
    from fetch3.optimize.run_optimization import run
    run(config_path)
