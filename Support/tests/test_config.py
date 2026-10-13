import pytest
from support.utils.config import validate_config


def test_valid(config):
    assert validate_config(config) is config


@pytest.mark.parametrize('section,key,value', [('attack','enabled',True),('model','input_dim',0),('federated','participation_rate',0),('training','learning_rate',float('nan')),('dataset','type','ciciot2023'),('model','num_classes',True)])
def test_invalid(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ValueError):
        validate_config(config)
