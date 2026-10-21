"""Verify model/data declarations with artificial metadata, without training."""

from dataclasses import FrozenInstanceError
import json

import pytest

from support.data.contract import ModelDataContract
from support.models.factory import create_model


@pytest.fixture
def contract_description():
    # These names are test placeholders, not a real dataset's feature/class map.
    return {
        'schema_version': 1,
        'model_type': 'mlp',
        'input_dim': 2,
        'num_classes': 2,
        'hidden_dims': [4],
        'feature_names': ['feature_a', 'feature_b'],
        'class_names': ['class_a', 'class_b'],
        'preprocessing_sha256': None,
    }


def test_roundtrip_and_model_factory_compatibility(contract_description):
    contract = ModelDataContract.from_dict(contract_description)
    serialized = json.dumps(contract.to_dict())
    restored = ModelDataContract.from_dict(json.loads(serialized))

    contract.assert_compatible(restored)
    assert restored == contract
    # Constructing a model verifies the factory contract; no training is run.
    model = create_model(contract.model_config())
    assert model.input_dim == 2
    assert model.num_classes == 2


@pytest.mark.parametrize('field,value', [
    ('model_type', 'cnn'),
    ('input_dim', 0),
    ('input_dim', True),
    ('num_classes', 1),
    ('num_classes', False),
    ('hidden_dims', []),
    ('hidden_dims', [0]),
    ('hidden_dims', [True]),
    ('hidden_dims', '4'),
    ('feature_names', []),
    ('feature_names', ['only_one']),
    ('feature_names', ['same', 'same']),
    ('feature_names', ['a', ' ']),
    ('feature_names', ['a', 1]),
    ('feature_names', ['a', ' b']),
    ('feature_names', 'ab'),
    ('class_names', ['same', 'same']),
    ('class_names', ['one']),
    ('class_names', [0, 1]),
    ('preprocessing_sha256', 'minmax'),
    ('preprocessing_sha256', ''),
    ('preprocessing_sha256', 'g' * 64),
    ('preprocessing_sha256', 42),
    ('schema_version', True),
    ('schema_version', 2),
])
def test_rejects_invalid_descriptions(contract_description, field, value):
    contract_description[field] = value

    with pytest.raises(ValueError):
        ModelDataContract.from_dict(contract_description)


@pytest.mark.parametrize('field', [
    'feature_names', 'class_names', 'preprocessing_sha256',
])
def test_missing_meanings_are_not_guessed(contract_description, field):
    del contract_description[field]

    with pytest.raises(ValueError):
        ModelDataContract.from_dict(contract_description)


def test_rejects_unknown_schema_fields(contract_description):
    contract_description['unexpected_setting'] = 1

    with pytest.raises(ValueError):
        ModelDataContract.from_dict(contract_description)


@pytest.mark.parametrize('field,value', [
    ('feature_names', ['feature_b', 'feature_a']),
    ('class_names', ['class_b', 'class_a']),
    ('hidden_dims', [5]),
    ('preprocessing_sha256', 'a' * 64),
])
def test_rejects_incompatible_pair(contract_description, field, value):
    original = ModelDataContract.from_dict(contract_description)
    contract_description[field] = value
    changed = ModelDataContract.from_dict(contract_description)

    # Equal feature/class counts do not make a changed order compatible.
    with pytest.raises(ValueError, match=field):
        original.assert_compatible(changed)


def test_rejects_different_fitted_preprocessing(contract_description):
    # Artificial hashes identify different declared artifacts; no scaler is fit.
    contract_description['preprocessing_sha256'] = 'a' * 64
    first = ModelDataContract.from_dict(contract_description)
    contract_description['preprocessing_sha256'] = 'b' * 64
    second = ModelDataContract.from_dict(contract_description)

    with pytest.raises(ValueError, match='preprocessing_sha256'):
        first.assert_compatible(second)


def test_caller_edits_cannot_change_contract(contract_description):
    contract = ModelDataContract.from_dict(contract_description)
    contract_description['feature_names'][0] = 'changed'
    contract_description['class_names'][0] = 'changed'
    contract_description['hidden_dims'][0] = 100
    exported = contract.to_dict()
    exported['feature_names'][0] = 'changed'
    model_config = contract.model_config()
    model_config['hidden_dims'][0] = 100

    assert contract.feature_names == ('feature_a', 'feature_b')
    assert contract.class_names == ('class_a', 'class_b')
    assert contract.hidden_dims == (4,)
    with pytest.raises(FrozenInstanceError):
        contract.input_dim = 10


def test_compatibility_requires_a_contract(contract_description):
    contract = ModelDataContract.from_dict(contract_description)

    with pytest.raises(TypeError):
        contract.assert_compatible(contract_description)


@pytest.mark.parametrize('description', [None, [], 1])
def test_rejects_non_mapping_descriptions(description):
    with pytest.raises(ValueError):
        ModelDataContract.from_dict(description)
