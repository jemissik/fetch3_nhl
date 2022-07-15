import yaml
from pathlib import Path

import pytest


def test_fetch3_nhl_run():
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config_files" / "model_config.yml"
    )
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent.parent / "output"

    with open(config_path, "r") as yml_config:
        loaded_configs = yaml.safe_load(yml_config)
        species_list = list(loaded_configs['species_parameters'].keys())

    from fetch3.main import main

    main(["--config_path", config_path, "--data_path", data_path, "--output_path", output_path], standalone_mode=False)


def test_fetch3_nhl_run_optconfig():
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config_files" / "opt_model_config.yml"
    )
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent.parent / "output"

    with open(config_path, "r") as yml_config:
        loaded_configs = yaml.safe_load(yml_config)
        species_list = list(loaded_configs['species_parameters'].keys())

    from fetch3.main import main

    main(["--config_path", config_path, "--data_path", data_path, "--output_path", output_path], standalone_mode=False)


def test_fetch3_PM_run():
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config_files" / "model_config_PM.yml"
    )
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent.parent / "output"

    with open(config_path, "r") as yml_config:
        loaded_configs = yaml.safe_load(yml_config)
        species_list = list(loaded_configs['species_parameters'].keys())

    from fetch3.main import main

    main(["--config_path", config_path, "--data_path", data_path, "--output_path", output_path], standalone_mode=False)


def test_fetch3_opt_run():
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config_files" / "opt_model_config.yml"
    )
    print(config_path)
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent.parent / "output"

    from fetch3.optimize.run_optimization import main
    main(["--config_file", config_path], standalone_mode=False)
    # main.callback(config_path)
