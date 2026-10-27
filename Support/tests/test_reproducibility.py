"""Verify seed reproducibility and accurate run-purpose metadata.

Metadata declarations do not establish research validity. The logger integration
case uses temporary synthetic arrays and fixed artificial weights without training.
"""

from datetime import datetime
import importlib.metadata
import json
import random
import subprocess

import numpy as np
import pytest
import torch
import yaml

import support.utils.reproducibility as reproducibility
from support.utils.reproducibility import environment_metadata, seed_everything


def test_seed_streams():
    seed_everything(42)
    a=(random.random(),np.random.rand(),torch.rand(3))
    seed_everything(42)
    b=(random.random(),np.random.rand(),torch.rand(3))
    assert a[:2]==b[:2]
    assert torch.equal(a[2],b[2])


@pytest.fixture
def git_queries(monkeypatch):
    """Simulate Git output without depending on the workspace history."""
    queries = []

    def read(command, **kwargs):
        queries.append((command, kwargs))
        if command[-2:] == ['rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        if command[-2:] == ['status', '--porcelain']:
            return ' M support/example.py\nA  support/new_example.py\n'
        raise AssertionError('Unexpected Git query')

    monkeypatch.setattr(reproducibility.subprocess, 'check_output', read)
    return queries


def test_original_call_keeps_dummy_defaults_and_records_environment(tmp_path, git_queries):
    result = environment_metadata(42, 'cpu', True, tmp_path)
    assert set(result) == {'seed', 'python', 'pytorch', 'numpy', 'scikit_learn', 'pyyaml',
                           'device', 'deterministic', 'timestamp_utc', 'git_commit',
                           'git_status', 'testing_only', 'purpose'}
    assert result['seed'] == 42 and result['device'] == 'cpu'
    assert result['deterministic'] is True
    assert result['testing_only'] is True
    assert result['purpose'] == 'Synthetic framework verification; not research results'
    assert result['git_commit'] == 'a' * 40
    assert result['git_status'] == ' M support/example.py\nA  support/new_example.py'
    for field in ['python', 'pytorch', 'numpy', 'scikit_learn', 'pyyaml']:
        assert isinstance(result[field], str) and result[field]
    stamp = datetime.fromisoformat(result['timestamp_utc'])
    assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('testing,purpose', [
    (True, 'Synthetic prepared-data prediction verification'),
    (False, 'Prepared-data evaluation'),
])
def test_explicit_declarations_are_preserved(tmp_path, git_queries, testing, purpose):
    result = environment_metadata(123, torch.device('cpu'), False, tmp_path,
                                  testing_only=testing, purpose=purpose)
    assert result['testing_only'] is testing
    assert result['purpose'] == purpose
    assert result['seed'] == 123
    assert result['device'] == 'cpu'
    assert result['deterministic'] is False


def test_non_testing_declaration_requires_purpose_before_queries(tmp_path, git_queries):
    with pytest.raises(ValueError, match='explicit purpose'):
        environment_metadata(42, 'cpu', True, tmp_path, testing_only=False)
    assert not git_queries


@pytest.mark.parametrize('value', [None, 0, 1, 'false', []])
def test_testing_flag_is_not_coerced(tmp_path, git_queries, value):
    with pytest.raises(ValueError, match='explicit boolean'):
        environment_metadata(42, 'cpu', True, tmp_path, testing_only=value, purpose='Example')
    assert not git_queries


@pytest.mark.parametrize('purpose', ['', ' ', ' leading', 'trailing ', 0, False, []])
def test_invalid_purposes_are_rejected_before_queries(tmp_path, git_queries, purpose):
    with pytest.raises(ValueError, match='purpose'):
        environment_metadata(42, 'cpu', True, tmp_path, purpose=purpose)
    assert not git_queries


def test_new_settings_are_keyword_only(tmp_path, git_queries):
    with pytest.raises(TypeError):
        environment_metadata(42, 'cpu', True, tmp_path, False, 'Prepared prediction')
    assert not git_queries


@pytest.mark.parametrize('failure', [OSError('git unavailable'), subprocess.CalledProcessError(128, ['git'])])
def test_git_failure_retains_other_metadata(tmp_path, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(reproducibility.subprocess, 'check_output', fail)
    result = environment_metadata(42, 'cpu', True, tmp_path)
    assert result['git_commit'] is None
    assert result['git_status'] is None
    assert result['testing_only'] is True
    assert result['python'] and result['pytorch']


def test_git_queries_keep_paths_with_spaces_as_individual_arguments(tmp_path, git_queries):
    repository = tmp_path / 'repo with spaces'
    environment_metadata(42, 'cpu', True, repository)
    assert len(git_queries) == 2
    for command, kwargs in git_queries:
        assert command[:5] == ['git', '-c', f'safe.directory={repository}', '-C', str(repository)]
        assert kwargs['text'] is True
        assert kwargs['stderr'] == subprocess.DEVNULL


def test_metadata_does_not_seed_change_settings_or_consume_rng(tmp_path, git_queries, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata collection must not seed or inspect CUDA availability')
    monkeypatch.setattr(reproducibility, 'seed_everything', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    environment_metadata(999, 'cuda:0', not before_deterministic, tmp_path,
                         testing_only=False, purpose='Prepared-data prediction')
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic


def test_missing_repository_does_not_create_files(tmp_path):
    repository = tmp_path / 'missing repo'
    result = environment_metadata(42, 'cpu', True, repository)
    assert result['git_commit'] is None and result['git_status'] is None
    assert list(tmp_path.iterdir()) == []


def test_required_package_errors_are_not_hidden(tmp_path, git_queries, monkeypatch):
    def fail(package):
        raise importlib.metadata.PackageNotFoundError(package)
    monkeypatch.setattr(reproducibility.importlib.metadata, 'version', fail)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        environment_metadata(42, 'cpu', True, tmp_path)


def test_metadata_integrates_with_logger_without_downgrading_testing_sources(tmp_path, git_queries):
    from support.data.contract import ModelDataContract
    from support.data.prepared import load_prepared_data
    from support.evaluation.predict import predict_prepared_data
    from support.models.artifact import load_model_artifact
    from support.models.factory import create_model
    from support.utils.parameters import clone_state
    from support.utils.prepared_config import load_prepared_config
    from support.utils.prepared_logger import save_prepared_run

    contract = ModelDataContract('mlp', 2, 2, (2,), ('a', 'b'), ('class_a', 'class_b'), None)
    model = create_model(contract.model_config())
    # Fixed weights test the metadata-to-logger path without fitting a model.
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(2))
        model.network[0].bias.zero_()
        model.network[2].weight.copy_(torch.eye(2))
        model.network[2].bias.zero_()
    torch.save({'schema_version': 1, 'model_name': 'test_fixture',
                'model_state': clone_state(model.state_dict()), 'contract': contract.to_dict(),
                'training_completed': False, 'testing_only': True}, tmp_path / 'model.pt')
    np.savez(tmp_path / 'data.npz', features=np.array([[2, 0], [0, 2]], dtype=np.float32),
             feature_names=np.array(contract.feature_names), class_names=np.array(contract.class_names))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'dataset_name': 'test_fixture', 'partition_name': 'prediction',
        'role': 'predict', 'client_id': None, 'data_format': 'npz', 'data_path': 'data.npz',
        'preprocessing_path': None, 'contract': contract.to_dict(), 'testing_only': True,
    }), encoding='utf-8')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'schema_version': 1, 'mode': 'predict',
        'experiment': {'name': 'metadata_test', 'seed': 42, 'device': 'cpu', 'deterministic': True, 'output_dir': 'runs'},
        'model': {'artifact_path': 'model.pt'},
        'dataset': {'manifest_path': 'manifest.json', 'expected_role': 'predict'},
        'inference': {'batch_size': 2},
    }), encoding='utf-8')
    config = load_prepared_config(tmp_path / 'config.yaml')
    artifact = load_model_artifact(tmp_path / 'model.pt')
    data = load_prepared_data(tmp_path / 'manifest.json', expected_role='predict')
    result = predict_prepared_data(artifact.model, data, model_contract=artifact.contract)
    # Intentionally conflicting environment declaration: the logger must retain
    # the testing-only model/data status. No non-testing research run is performed.
    metadata = environment_metadata(42, 'cpu', True, tmp_path, testing_only=False, purpose='Prepared-data prediction')
    directory = save_prepared_run(config, artifact, data, result, metadata=metadata)
    record = json.loads((directory / 'metadata.json').read_text())
    assert record['environment']['testing_only'] is False
    assert record['environment']['purpose'] == 'Prepared-data prediction'
    assert record['testing_only'] is True
    assert json.loads((directory / 'run_summary.json').read_text())['testing_only'] is True
