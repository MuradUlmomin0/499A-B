"""Prepared logger checks using temporary files and fixed artificial weights.

Synthetic fixtures verify software behavior only; no optimizer or model training
is used. Purpose flags exercise declarations, not research validity.
"""

from copy import deepcopy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from support.data.contract import ModelDataContract
from support.data.prepared import load_prepared_data
from support.evaluation.predict import predict_prepared_data
from support.evaluation.prepared import evaluate_prepared_data
from support.models.artifact import load_model_artifact
from support.models.factory import create_model
import support.utils.prepared_logger as logger_module
from support.utils.prepared_logger import save_prepared_run
from support.utils.prepared_config import load_prepared_config
from support.utils.parameters import clone_state


@pytest.fixture
def run_factory(tmp_path):
    """Return a factory for complete prediction/evaluation fixtures."""
    def make(mode='predict'):
        folder = tmp_path / mode
        folder.mkdir(exist_ok=True)
        contract = ModelDataContract(
            model_type='mlp', input_dim=2, num_classes=2, hidden_dims=(2,),
            feature_names=('feature_a', 'feature_b'), class_names=('class_a', 'class_b'),
            preprocessing_sha256=None,
        )
        model = create_model(contract.model_config())
        # Hand-set weights give known answers without fitting a model.
        with torch.no_grad():
            model.network[0].weight.copy_(torch.eye(2))
            model.network[0].bias.zero_()
            model.network[2].weight.copy_(2 * torch.eye(2))
            model.network[2].bias.zero_()
        torch.save({
            'schema_version': 1, 'model_name': 'fixed_test_fixture',
            'model_state': clone_state(model.state_dict()), 'contract': contract.to_dict(),
            'training_completed': False, 'testing_only': True,
        }, folder / 'model.pt')
        artifact = load_model_artifact(folder / 'model.pt')
        arrays = {
            'features': np.array([[3, 1], [1, 4], [2, 2], [0, 3], [5, 0]], dtype=np.float32),
            'feature_names': np.array(contract.feature_names),
            'class_names': np.array(contract.class_names),
        }
        if mode == 'evaluate':
            arrays['labels'] = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        np.savez(folder / 'data.npz', **arrays)
        role = 'test' if mode == 'evaluate' else 'predict'
        (folder / 'manifest.json').write_text(json.dumps({
            'schema_version': 1, 'dataset_name': 'fixed_test_fixture', 'partition_name': role,
            'role': role, 'client_id': None, 'data_format': 'npz', 'data_path': 'data.npz',
            'preprocessing_path': None, 'contract': contract.to_dict(), 'testing_only': True,
        }), encoding='utf-8')
        data = load_prepared_data(folder / 'manifest.json', expected_role=role, expected_contract=contract)
        raw_config = {
            'schema_version': 1, 'mode': mode,
            'experiment': {'name': 'prepared_test', 'seed': 42, 'device': 'cpu',
                           'deterministic': True, 'output_dir': '../runs'},
            'model': {'artifact_path': 'model.pt'},
            'dataset': {'manifest_path': 'manifest.json', 'expected_role': role},
            'inference': {'batch_size': 2},
        }
        (folder / 'config.yaml').write_text(yaml.safe_dump(raw_config), encoding='utf-8')
        config = load_prepared_config(folder / 'config.yaml')
        if mode == 'predict':
            result = predict_prepared_data(artifact.model, data, model_contract=contract, batch_size=2)
        else:
            result = evaluate_prepared_data(artifact, data, expected_role=role, batch_size=2)
        metadata = {
            'seed': 42, 'python': platform.python_version(), 'pytorch': str(torch.__version__),
            'device': 'cpu', 'deterministic': True, 'testing_only': True,
            'git_commit': None, 'purpose': 'Synthetic software verification',
        }
        return config, artifact, data, result, metadata
    return make


def log_run(run):
    """Pass an already-computed fixture result to the public logger API."""
    config, artifact, data, result, metadata = run
    return save_prepared_run(config, artifact, data, result, metadata=metadata)


def assert_no_outputs(config):
    """Invalid inputs must be rejected before creating the output root."""
    assert not Path(config['experiment']['output_dir']).exists()


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_complete_run_files_and_source_records(run_factory, mode):
    run = run_factory(mode)
    config, artifact, data, result, metadata = run
    directory = log_run(run)
    output = 'predictions.csv' if mode == 'predict' else 'final_metrics.json'
    assert directory.is_absolute()
    assert directory.parent == Path(config['experiment']['output_dir'])
    assert {path.name for path in directory.iterdir()} == {'config.yaml', 'metadata.json', output, 'run_summary.json'}
    assert yaml.safe_load((directory / 'config.yaml').read_text()) == config
    record = json.loads((directory / 'metadata.json').read_text())
    assert record['testing_only'] is True
    assert record['environment'] == metadata
    assert record['model']['artifact_sha256'] == artifact.artifact_sha256
    assert record['model']['contract'] == artifact.contract.to_dict()
    assert record['dataset']['contract'] == data.manifest.contract.to_dict()
    assert record['dataset']['manifest_path'] == str(data.manifest.manifest_path)
    assert record['dataset']['data_path'] == str(data.manifest.data_path)
    assert record['dataset']['has_labels'] is (mode == 'evaluate')
    assert record['dataset']['sample_count'] == 5
    assert datetime.fromisoformat(record['created_at_utc']).utcoffset().total_seconds() == 0
    summary = json.loads((directory / 'run_summary.json').read_text())
    assert summary == {'schema_version': 1, 'status': 'completed', 'mode': mode,
                       'testing_only': True, 'sample_count': 5, 'output_file': output}
    if mode == 'predict':
        with (directory / output).open(newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert reader.fieldnames == ['row_index', 'class_id', 'class_name', 'probability_0', 'probability_1', 'testing_only']
        assert [int(row['row_index']) for row in rows] == list(range(5))
        assert [int(row['class_id']) for row in rows] == [0, 1, 0, 1, 0]
        assert [row['class_name'] for row in rows] == ['class_a', 'class_b', 'class_a', 'class_b', 'class_a']
        assert all(row['testing_only'] == 'True' for row in rows)
        assert np.allclose([[float(row[f'probability_{i}']) for i in range(2)] for row in rows], result.probabilities.numpy())
    else:
        output_record = json.loads((directory / output).read_text())
        assert output_record['metrics'] == result.metrics
        assert output_record['metrics']['accuracy'] == pytest.approx(0.8)
        assert output_record['testing_only'] is True
        assert output_record['artifact_sha256'] == artifact.artifact_sha256
        assert output_record['class_names'] == ['class_a', 'class_b']


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
@pytest.mark.parametrize('model_testing,data_testing,environment_testing', [
    (False, False, False), (False, False, True), (False, True, False), (False, True, True),
    (True, False, False), (True, False, True), (True, True, False), (True, True, True),
])
def test_purpose_is_conservative_across_all_inputs(run_factory, mode, model_testing, data_testing, environment_testing):
    # Metadata declarations only; these fixtures are not trained or research results.
    config, artifact, data, result, metadata = run_factory(mode)
    artifact = replace(artifact, training_completed=True, testing_only=model_testing)
    data = replace(data, manifest=replace(data.manifest, testing_only=data_testing))
    if mode == 'evaluate':
        result = replace(result, manifest=data.manifest, training_completed=True,
                         testing_only=model_testing or data_testing)
    else:
        result = replace(result, manifest=data.manifest)
    metadata['testing_only'] = environment_testing
    directory = log_run((config, artifact, data, result, metadata))
    expected = model_testing or data_testing or environment_testing
    assert json.loads((directory / 'metadata.json').read_text())['testing_only'] is expected
    assert json.loads((directory / 'run_summary.json').read_text())['testing_only'] is expected
    if mode == 'evaluate':
        assert json.loads((directory / 'final_metrics.json').read_text())['testing_only'] is expected
    else:
        with (directory / 'predictions.csv').open(newline='') as stream:
            assert all(row['testing_only'] == str(expected) for row in csv.DictReader(stream))


def test_environment_flag_is_optional_and_cannot_replace_source_records(run_factory):
    run = list(run_factory())
    del run[-1]['testing_only']
    run[-1]['model'] = {'artifact_sha256': 'caller cannot replace source'}
    record = json.loads((log_run(run) / 'metadata.json').read_text())
    assert record['testing_only'] is True
    assert record['model']['artifact_sha256'] == run[1].artifact_sha256
    assert record['environment']['model'] == run[-1]['model']


def test_repeated_saves_keep_previous_runs_unchanged(run_factory):
    run = run_factory()
    first = log_run(run)
    snapshot = {p.name: p.read_bytes() for p in first.iterdir()}
    second = log_run(run)
    assert first != second
    assert {p.name: p.read_bytes() for p in first.iterdir()} == snapshot


def test_collision_fails_without_overwriting_existing_run(run_factory, monkeypatch):
    run = run_factory()
    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    class FixedClock:
        @staticmethod
        def now(tz):
            return fixed_time
    monkeypatch.setattr(logger_module, 'datetime', FixedClock)
    monkeypatch.setattr(logger_module, 'uuid4', lambda: SimpleNamespace(hex='0' * 32))
    directory = log_run(run)
    snapshot = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        log_run(run)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == snapshot


@pytest.mark.parametrize('field,value', [('seed', True), ('seed', 43), ('device', 'cuda'),
                                         ('deterministic', 1), ('deterministic', False),
                                         ('python', ''), ('pytorch', None), ('testing_only', 'false')])
def test_invalid_execution_metadata_does_not_create_outputs(run_factory, field, value):
    run = list(run_factory())
    run[-1][field] = value
    with pytest.raises(ValueError):
        log_run(run)
    assert_no_outputs(run[0])


@pytest.mark.parametrize('problem', ['not_dict', 'missing_field', 'unsupported_value', 'nonfinite'])
def test_metadata_must_be_complete_and_serializable(run_factory, problem):
    run = list(run_factory())
    if problem == 'not_dict':
        run[-1] = []
    elif problem == 'missing_field':
        del run[-1]['seed']
    elif problem == 'unsupported_value':
        run[-1]['extra'] = object()
    else:
        run[-1]['extra'] = float('nan')
    with pytest.raises((TypeError, ValueError)):
        log_run(run)
    assert_no_outputs(run[0])


@pytest.mark.parametrize('field', ['output_dir', 'artifact_path', 'manifest_path'])
def test_requires_absolute_normalized_configuration_paths(run_factory, tmp_path, monkeypatch, field):
    run = list(run_factory())
    monkeypatch.chdir(tmp_path)
    section = {'output_dir': 'experiment', 'artifact_path': 'model', 'manifest_path': 'dataset'}[field]
    run[0][section][field] = 'relative/path'
    with pytest.raises(ValueError, match='absolute paths'):
        log_run(run)
    # Keep possible accidental relative writes inside pytest's temporary folder.
    assert_no_outputs(run[0])
    assert not Path('relative/path').exists()


@pytest.mark.parametrize('problem', ['model_path', 'manifest_path', 'role', 'contract', 'result_manifest', 'sha'])
def test_mixed_sources_are_rejected_before_writes(run_factory, problem):
    config, artifact, data, result, metadata = run_factory()
    if problem == 'model_path':
        config['model']['artifact_path'] += '.other'
    elif problem == 'manifest_path':
        config['dataset']['manifest_path'] += '.other'
    elif problem == 'role':
        config['dataset']['expected_role'] = 'test'
    elif problem == 'contract':
        description = data.manifest.contract.to_dict()
        description['class_names'].reverse()
        data = replace(data, manifest=replace(data.manifest, contract=ModelDataContract.from_dict(description)))
    elif problem == 'result_manifest':
        result = replace(result, manifest=replace(result.manifest, partition_name='other'))
    else:
        artifact = replace(artifact, artifact_sha256='not-a-sha')
    with pytest.raises(ValueError):
        log_run((config, artifact, data, result, metadata))
    assert_no_outputs(config)


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_mode_must_match_result_type(run_factory, mode):
    run = list(run_factory(mode))
    run[0]['mode'] = 'evaluate' if mode == 'predict' else 'predict'
    with pytest.raises(TypeError, match='Result type'):
        log_run(run)
    assert_no_outputs(run[0])


@pytest.mark.parametrize('field,value', [('model_name', 'other'), ('artifact_path', Path('/tmp/other.pt')),
                                        ('artifact_sha256', 'a' * 64), ('sample_count', 4),
                                        ('sample_count', True), ('training_completed', True), ('testing_only', False)])
def test_evaluation_metadata_must_match_its_inputs(run_factory, field, value):
    run = list(run_factory('evaluate'))
    run[3] = replace(run[3], **{field: value})
    with pytest.raises(ValueError):
        log_run(run)
    assert_no_outputs(run[0])


@pytest.mark.parametrize('problem', ['empty', 'nan', 'infinity', 'unsupported'])
def test_bad_metrics_are_rejected_before_writes(run_factory, problem):
    run = list(run_factory('evaluate'))
    metrics = {} if problem == 'empty' else dict(run[3].metrics)
    if problem == 'nan':
        metrics['loss'] = float('nan')
    elif problem == 'infinity':
        metrics['loss'] = float('inf')
    elif problem == 'unsupported':
        metrics['loss'] = object()
    run[3] = replace(run[3], metrics=metrics)
    with pytest.raises((ValueError, TypeError)):
        log_run(run)
    assert_no_outputs(run[0])


@pytest.mark.parametrize('problem', ['id_dtype', 'id_shape', 'negative_id', 'large_id', 'id_device',
                                    'prob_nan', 'prob_inf', 'prob_negative', 'prob_large', 'prob_shape',
                                    'prob_integer', 'prob_device', 'prob_sum'])
def test_bad_predictions_are_rejected_before_writes(run_factory, problem):
    run = list(run_factory())
    ids, probabilities = run[3].class_ids.clone(), run[3].probabilities.clone()
    if problem == 'id_dtype':
        ids = ids.float()
    elif problem == 'id_shape':
        ids = ids.unsqueeze(1)
    elif problem == 'negative_id':
        ids[0] = -1
    elif problem == 'large_id':
        ids[0] = 2
    elif problem == 'id_device':
        ids = ids.to('meta')
    elif problem == 'prob_nan':
        probabilities[0, 0] = float('nan')
    elif problem == 'prob_inf':
        probabilities[0, 0] = float('inf')
    elif problem == 'prob_negative':
        probabilities[0, 0] = -0.1
    elif problem == 'prob_large':
        probabilities[0, 0] = 1.1
    elif problem == 'prob_shape':
        probabilities = probabilities[:, :1]
    elif problem == 'prob_integer':
        probabilities = probabilities.long()
    elif problem == 'prob_device':
        probabilities = probabilities.to('meta')
    else:
        probabilities[0] = 0
    run[3] = replace(run[3], class_ids=ids, probabilities=probabilities)
    with pytest.raises(ValueError):
        log_run(run)
    assert_no_outputs(run[0])


def test_softmax_ties_do_not_replace_supplied_class_decisions(run_factory):
    run = list(run_factory())
    run[3] = replace(run[3], class_ids=torch.ones(5, dtype=torch.long), probabilities=torch.full((5, 2), 0.5))
    with (log_run(run) / 'predictions.csv').open(newline='') as stream:
        assert [int(row['class_id']) for row in csv.DictReader(stream)] == [1] * 5


def test_write_failure_leaves_no_completed_summary(run_factory, monkeypatch):
    run = run_factory()
    original_open = Path.open
    def fail_output(path, *args, **kwargs):
        if path.name == 'predictions.csv':
            raise OSError('simulated output failure')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_output)
    with pytest.raises(OSError, match='simulated'):
        log_run(run)
    folders = list(Path(run[0]['experiment']['output_dir']).iterdir())
    assert len(folders) == 1
    assert (folders[0] / 'config.yaml').exists()
    assert not (folders[0] / 'run_summary.json').exists()


def test_inputs_models_and_rng_are_preserved(run_factory):
    run = run_factory()
    config, artifact, data, result, metadata = run
    config_before, metadata_before = deepcopy(config), deepcopy(metadata)
    parameters = clone_state(artifact.model.state_dict())
    modes = [module.training for module in artifact.model.modules()]
    tensors = [tensor.clone() for tensor in data.dataset.tensors]
    ids, probabilities = result.class_ids.clone(), result.probabilities.clone()
    torch_before = torch.get_rng_state().clone()
    random_before, numpy_before = random.getstate(), np.random.get_state()
    log_run(run)
    assert config == config_before and metadata == metadata_before
    assert [module.training for module in artifact.model.modules()] == modes
    assert all(torch.equal(value, parameters[key]) for key, value in artifact.model.state_dict().items())
    assert all(torch.equal(before, after) for before, after in zip(tensors, data.dataset.tensors))
    assert torch.equal(ids, result.class_ids) and torch.equal(probabilities, result.probabilities)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert random_before == random.getstate()
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
