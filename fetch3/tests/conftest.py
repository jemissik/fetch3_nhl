from pathlib import Path


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
