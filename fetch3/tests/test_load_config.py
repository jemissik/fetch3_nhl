import pytest

from fetch3.model_config import config_from_groupers, ConfigParams


def test_config_from_groupers(param_groups_config):
    configs = config_from_groupers(param_groups_config)
    for config in configs:
        assert config.species in ["oak",  "maple"]


@pytest.mark.parametrize(
    "config",
    ["model_config", "model_pm_config", "opt_model_config",]  # 1. pass fixture name as a string
)
def test_deprecated_config(config, request):
    config = request.getfixturevalue(config)
    ConfigParams.from_deprecated_config(config=config)
