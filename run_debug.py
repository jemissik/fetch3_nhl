from pathlib import Path
from fetch3.__main__ import run


parent_path = Path(__file__).resolve().parent
default_config_path = parent_path / 'config_files' / 'model_config.yml'
default_data_path = parent_path / 'data'
default_output_path = parent_path / 'output'

run(default_config_path, default_data_path, default_output_path)