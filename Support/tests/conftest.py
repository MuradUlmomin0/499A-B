import pytest
from pathlib import Path
from support.utils.config import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / 'configs/fedavg_dummy_test.yaml')
