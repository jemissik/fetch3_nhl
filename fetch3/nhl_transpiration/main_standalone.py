"""
This file runs NHL as a standalone module (without running FETCH3).

It returns NHL transpiration, and also writes NHL
transpiration to a netcdf file.

"""


import fetch3.nhl_transpiration.main as nhl
import click
from pathlib import Path
from fetch3.model_config import setup_config

config_file = Path('/Users/jmissik/Desktop/repos.nosync/fetch3_nhl/config_files/opt_model_config.yml')
output_dir = Path('/Users/jmissik/Desktop/repos.nosync/fetch3_nhl/output/nhl_test')

parent_path = Path(__file__).resolve().parent.parent.parent
default_config_path = parent_path / "config_files" / "model_config.yml"
default_data_path = parent_path / "data"
default_output_path = parent_path / "output"
model_dir = Path(__file__).parent.resolve()  # File path of model source code

@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(default_config_path),
    help="Path to configuration YAML file",
)
@click.option(
    "--data_path",
    type=click.Path(exists=True, path_type=Path),
    default=str(default_data_path),
    help="Path to data directory",
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, path_type=Path),
    default=str(parent_path),
    help="Path to output directory",
)
@click.option(
    "--species",
    type=str,
    default=None,
    help="species to run the model for"
)
def run(config_path, data_path, output_path, species):
    cfg = setup_config(config_path, species=species)

    # NHL in units of [kg H2O m-2crown_projection s-1 m-1stem]
    nhl_trans_tot, LAD = nhl.main(cfg, output_path, data_path, to_model_res=False, write_output=True)


if __name__ == "__main__":
    run()