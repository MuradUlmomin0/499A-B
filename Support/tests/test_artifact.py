"""Model artifact verification using fixed artificial weights; no training.

All files and synthetic arrays are temporary, testing-only fixtures. Producer
purpose declarations are exercised as metadata, not evidence of training or
research validity.
"""

from collections import OrderedDict
import hashlib
import json
import pickle

import numpy as np
import pytest
import torch

from support.data.contract import ModelDataContract
from support.data.prepared import load_prepared_data
from support.evaluation.predict import predict_prepared_data
from support.models.artifact import ModelArtifact, load_model_artifact
from support.models.checkpoint import load_model_checkpoint
from support.models.factory import create_model
from support.utils.parameters import clone_state


@pytest.fixture
def artifact_fixture(tmp_path):
    contract = ModelDataContract(
        model_type='mlp', input_dim=2, num_classes=2, hidden_dims=(2,),
        feature_names=('feature_a', 'feature_b'),
        class_names=('class_a', 'class_b'), preprocessing_sha256=None,
    )
    model = create_model(contract.model_config())
    # On nonnegative inputs, these hand-set weights give logits 2 * features.
    # Known answers let us verify loading and inference without fitting a model.
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(2))
        model.network[0].bias.zero_()
        model.network[2].weight.copy_(2 * torch.eye(2))
        model.network[2].bias.zero_()
    description = {
        'schema_version': 1,
        'model_name': 'fixed_test_fixture',
        'model_state': clone_state(model.state_dict()),
        'contract': contract.to_dict(),
        'training_completed': False,
        'testing_only': True,
    }
    return tmp_path / 'model.pt', model, contract, description


def save_and_load(path, description):
    """Serialize a test description, then exercise the public artifact loader."""
    torch.save(description, path)
    return load_model_artifact(path)


def test_restores_weights_metadata_and_preserves_source(artifact_fixture):
    path, original, contract, description = artifact_fixture
    torch.save(description, path)
    content = path.read_bytes()
    before = clone_state(original.state_dict())

    artifact = load_model_artifact(path)

    assert isinstance(artifact, ModelArtifact)
    assert artifact.contract == contract
    assert artifact.artifact_path == path.resolve()
    assert artifact.artifact_sha256 == hashlib.sha256(content).hexdigest()
    assert artifact.model_name == 'fixed_test_fixture'
    assert artifact.training_completed is False
    assert artifact.testing_only is True
    assert artifact.model is not original
    assert all(not module.training for module in artifact.model.modules())
    assert all(parameter.device.type == 'cpu' for parameter in artifact.model.parameters())
    for name, saved in description['model_state'].items():
        assert torch.equal(artifact.model.state_dict()[name], saved)
        assert torch.equal(original.state_dict()[name], before[name])
    assert path.read_bytes() == content


def test_accepts_reordered_state_keys(artifact_fixture):
    path, model, contract, description = artifact_fixture
    description['model_state'] = OrderedDict(reversed(list(description['model_state'].items())))

    artifact = save_and_load(path, description)

    assert list(artifact.model.state_dict()) == list(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in artifact.model.state_dict().items())


def test_artifact_weights_work_with_existing_checkpoint_loader(artifact_fixture):
    path, model, contract, description = artifact_fixture
    torch.save(description, path)
    target = create_model(contract.model_config())

    returned = load_model_checkpoint(target, path)

    assert returned is target
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in target.state_dict().items())


def test_repeated_loading_preserves_rng_and_restores_identical_weights(artifact_fixture):
    path, model, contract, description = artifact_fixture
    torch.save(description, path)
    before = torch.get_rng_state().clone()

    first = load_model_artifact(path)
    second = load_model_artifact(path)

    assert torch.equal(before, torch.get_rng_state())
    assert first.artifact_sha256 == second.artifact_sha256
    assert all(torch.equal(value, second.model.state_dict()[name])
               for name, value in first.model.state_dict().items())


def test_independent_loads_do_not_share_parameter_storage(artifact_fixture):
    path, model, contract, description = artifact_fixture
    torch.save(description, path)
    first = load_model_artifact(path)
    second = load_model_artifact(path)
    expected = clone_state(second.model.state_dict())
    content = path.read_bytes()

    with torch.no_grad():
        for parameter in first.model.parameters():
            parameter.add_(10)

    assert all(torch.equal(value, expected[name])
               for name, value in second.model.state_dict().items())
    assert all(torch.equal(value, expected[name]) for name, value in model.state_dict().items())
    assert path.read_bytes() == content


def test_fingerprint_tracks_changed_artifact_contents(artifact_fixture):
    path, model, contract, description = artifact_fixture
    first = save_and_load(path, description)
    description['model_state']['network.2.bias'][0] = 3

    second = save_and_load(path, description)

    assert first.artifact_sha256 != second.artifact_sha256
    assert second.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.model.network[2].bias[0].item() == 0
    assert second.model.network[2].bias[0].item() == 3


def test_failed_parameter_validation_preserves_rng(artifact_fixture):
    path, model, contract, description = artifact_fixture
    description['model_state']['network.0.weight'] = torch.zeros(7, 7)
    torch.save(description, path)
    before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match='Shape or dtype'):
        load_model_artifact(path)

    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('completed,testing', [(False, True), (True, True), (True, False)])
def test_preserves_explicit_purpose_declarations(artifact_fixture, completed, testing):
    # True is a producer claim in these schema cases; no actual training occurs.
    path, model, contract, description = artifact_fixture
    description.update(training_completed=completed, testing_only=testing)

    artifact = save_and_load(path, description)

    assert artifact.training_completed is completed
    assert artifact.testing_only is testing


def test_declared_untrained_model_must_remain_testing_only(artifact_fixture):
    path, model, contract, description = artifact_fixture
    description['testing_only'] = False

    with pytest.raises(ValueError, match='untrained'):
        save_and_load(path, description)


@pytest.mark.parametrize('field', [
    'schema_version', 'model_name', 'model_state', 'contract',
    'training_completed', 'testing_only',
])
def test_rejects_missing_required_fields(artifact_fixture, field):
    path, model, contract, description = artifact_fixture
    del description[field]

    with pytest.raises(ValueError, match='fields'):
        save_and_load(path, description)


def test_rejects_unknown_fields(artifact_fixture):
    path, model, contract, description = artifact_fixture
    description['extra'] = 5

    with pytest.raises(ValueError, match='fields'):
        save_and_load(path, description)


@pytest.mark.parametrize('description', [None, [], 5])
def test_rejects_nonmapping_artifacts(tmp_path, description):
    with pytest.raises(ValueError, match='fields'):
        save_and_load(tmp_path / 'invalid.pt', description)


@pytest.mark.parametrize('version', [True, 0, 2, '1'])
def test_rejects_invalid_schema_versions(artifact_fixture, version):
    path, model, contract, description = artifact_fixture
    description['schema_version'] = version

    with pytest.raises(ValueError, match='version'):
        save_and_load(path, description)


@pytest.mark.parametrize('name', ['', '  ', ' fixture', 'fixture ', 42])
def test_rejects_invalid_model_names(artifact_fixture, name):
    path, model, contract, description = artifact_fixture
    description['model_name'] = name

    with pytest.raises(ValueError, match='model_name'):
        save_and_load(path, description)


@pytest.mark.parametrize('field,value', [
    ('training_completed', 1), ('training_completed', 'false'),
    ('testing_only', 0), ('testing_only', 'true'),
])
def test_requires_boolean_purpose_flags(artifact_fixture, field, value):
    path, model, contract, description = artifact_fixture
    description[field] = value

    with pytest.raises(ValueError, match='boolean'):
        save_and_load(path, description)


@pytest.mark.parametrize('problem', [
    'missing', 'extra', 'shape', 'dtype', 'nan', 'infinity',
    'nontensor', 'empty', 'list',
])
def test_rejects_invalid_weights(artifact_fixture, problem):
    path, model, contract, description = artifact_fixture
    state = description['model_state']
    key = next(iter(state))
    if problem == 'missing':
        del state[key]
    elif problem == 'extra':
        state['extra'] = torch.zeros(1)
    elif problem == 'shape':
        state[key] = torch.zeros(9)
    elif problem == 'dtype':
        state[key] = state[key].double()
    elif problem in ('nan', 'infinity'):
        state[key] = torch.full_like(state[key], float('nan' if problem == 'nan' else 'inf'))
    elif problem == 'nontensor':
        state[key] = 'weights'
    elif problem == 'empty':
        description['model_state'] = {}
    else:
        description['model_state'] = list(state.values())

    with pytest.raises(ValueError):
        save_and_load(path, description)


@pytest.mark.parametrize('problem', ['hidden_dims', 'classes', 'feature_names', 'model_type'])
def test_rejects_invalid_contracts_or_architecture_mismatches(artifact_fixture, problem):
    path, model, contract, description = artifact_fixture
    if problem == 'hidden_dims':
        description['contract']['hidden_dims'] = [3]
    elif problem == 'classes':
        description['contract'].update(num_classes=3, class_names=['class_a', 'class_b', 'class_c'])
    elif problem == 'feature_names':
        description['contract']['feature_names'] = ['feature_a', 'feature_a']
    else:
        description['contract']['model_type'] = 'unsupported'

    with pytest.raises(ValueError):
        save_and_load(path, description)


@pytest.mark.parametrize('kind', ['missing', 'directory'])
def test_requires_an_existing_file(tmp_path, kind):
    path = tmp_path / 'model.pt'
    if kind == 'directory':
        path.mkdir()

    with pytest.raises(FileNotFoundError):
        load_model_artifact(path)


def test_rejects_corrupt_files(tmp_path):
    path = tmp_path / 'corrupt.pt'
    path.write_bytes(b'not a torch artifact')

    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
        load_model_artifact(path)


def test_restricted_reader_rejects_full_pickled_models(artifact_fixture):
    path, model, contract, description = artifact_fixture
    torch.save(model, path)

    with pytest.raises(pickle.UnpicklingError):
        load_model_artifact(path)


def test_loaded_artifact_predicts_known_answers_without_training(artifact_fixture):
    path, model, contract, description = artifact_fixture
    artifact = save_and_load(path, description)
    np.savez(
        path.parent / 'fixture.npz',
        features=np.array([[3, 1], [1, 4], [2, 2]], dtype=np.float32),
        feature_names=np.array(contract.feature_names),
        class_names=np.array(contract.class_names),
    )
    manifest = path.parent / 'manifest.json'
    manifest.write_text(json.dumps({
        'schema_version': 1, 'dataset_name': 'fixed_test_fixture',
        'partition_name': 'prediction', 'role': 'predict', 'client_id': None,
        'data_format': 'npz', 'data_path': 'fixture.npz',
        'preprocessing_path': None, 'contract': contract.to_dict(), 'testing_only': True,
    }), encoding='utf-8')
    data = load_prepared_data(manifest, expected_role='predict', expected_contract=artifact.contract)

    predictions = predict_prepared_data(
        artifact.model, data, model_contract=artifact.contract, batch_size=2,
    )

    assert predictions.class_ids.tolist() == [0, 1, 0]
    assert predictions.predicted_labels == ('class_a', 'class_b', 'class_a')
    assert artifact.testing_only is True
    assert predictions.manifest.testing_only is True


@pytest.mark.parametrize('field', ['feature_names', 'class_names'])
def test_model_contract_remains_independent_of_dataset_declarations(artifact_fixture, field):
    path, model, contract, description = artifact_fixture
    artifact = save_and_load(path, description)
    dataset_description = contract.to_dict()
    dataset_description[field].reverse()
    dataset_contract = ModelDataContract.from_dict(dataset_description)

    # Equal shapes cannot detect swapped column meanings or class IDs. The
    # independently saved model contract must reject these dataset declarations.
    with pytest.raises(ValueError, match=field):
        artifact.contract.assert_compatible(dataset_contract)

    assert artifact.contract == contract
