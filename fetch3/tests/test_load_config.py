import pathlib
from fetch3.model_config import config_from_groupers


def test_config_from_groupers(param_groups_config):
    configs = config_from_groupers(param_groups_config)
    for config in configs:
        assert config.species in ["oak",  "maple"]
