from pathlib import Path
import pytest
from fetch3.utils import load_yaml

TEST_CONFIGS = Path(__file__).parent.parent.parent / "config_files"


def pytest_addoption(parser):
    parser.addoption(
        "--config_path",
        action="store",
        default=Path(__file__).resolve().parent.parent.parent / "config_files" / "test_nhl_oak.yml",
    )
    parser.addoption(
        "--data_path",
        action="store",
        default=Path(__file__).resolve().parent.parent.parent / "data",
    )
    parser.addoption(
        "--output_path", action="store", default=Path(__file__).resolve().parent / "output"
    )


@pytest.fixture
def param_groups_config():
    config_path = TEST_CONFIGS / "test_param_groups.yml"
    return load_yaml(config_path)


@pytest.fixture
def model_config():
    config_path = TEST_CONFIGS / "model_config.yml"
    return load_yaml(config_path)


@pytest.fixture
def model_pm_config():
    config_path = TEST_CONFIGS / "model_config_PM.yml"
    return load_yaml(config_path)


@pytest.fixture
def opt_model_config():
    config_path = TEST_CONFIGS / "opt_model_config.yml"
    return load_yaml(config_path)

