"""Verify complete prepared runs with synthetic data and fixed artificial weights.

No model is trained. Purpose-flag cases exercise metadata declarations only;
neither those declarations nor the toy metrics establish research validity.
All run inputs and outputs are created in pytest's temporary directories.
"""

import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from support.data.contract import ModelDataContract
from support.models.factory import create_model
import support.prepared_runner as runner
from support.utils.parameters import clone_state
import support.utils.reproducibility as reproducibility


@pytest.fixture(autouse=True)
def isolate_process_state(monkeypatch):
    """Restore RNG/backend settings and keep verification independent of Git/GPU."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')

    # These tests execute on CPU. GPU execution needs separate hardware checks.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def git_query(command, **kwargs):
        if command[-2:] == ['rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        if command[-2:] == ['status', '--porcelain']:
            return ''
        raise AssertionError('Unexpected subprocess request')

    monkeypatch.setattr(reproducibility.subprocess, 'check_output', git_query)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_state
        if previous_workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = previous_workspace


@pytest.fixture
def artifacts(tmp_path):
    """Build complete saved-model, data, manifest and YAML input fixtures."""
    def make(
        mode='predict', *, role=None, labeled=None, model_testing=True,
        data_testing=True, input_dim=2, num_classes=2, deterministic=True,
    ):
        folder = tmp_path / mode
        folder.mkdir(exist_ok=True)
        if role is None:
            role = 'predict' if mode == 'predict' else 'test'
        if labeled is None:
            labeled = mode == 'evaluate' or role != 'predict'
        contract = ModelDataContract(
            model_type='mlp', input_dim=input_dim, num_classes=num_classes,
            hidden_dims=(input_dim,),
            feature_names=tuple(f'feature_{i}' for i in range(input_dim)),
            class_names=tuple(f'class_{i}' for i in range(num_classes)),
            preprocessing_sha256=None,
        )
        model = create_model(contract.model_config())
        # Identity hidden weights and a diagonal output layer give known answers.
        # This assigns weights directly; there is no fitting step.
        with torch.no_grad():
            model.network[0].weight.copy_(torch.eye(input_dim))
            model.network[0].bias.zero_()
            model.network[2].weight.zero_()
            for index in range(min(input_dim, num_classes)):
                model.network[2].weight[index, index] = 2
            model.network[2].bias.zero_()
        torch.save({
            'schema_version': 1, 'model_name': 'fixed_test_fixture',
            'model_state': clone_state(model.state_dict()), 'contract': contract.to_dict(),
            # False-testing cases simulate a producer's declarations so the OR
            # rule can be checked. These artificial weights were never trained.
            'training_completed': not model_testing, 'testing_only': model_testing,
        }, folder / 'model.pt')
        features = np.zeros((5, input_dim), dtype=np.float32)
        features[:, :2] = np.array(
            [[3, 1], [1, 4], [2, 2], [0, 3], [5, 0]], dtype=np.float32,
        )
        arrays = {
            'features': features, 'feature_names': np.array(contract.feature_names),
            'class_names': np.array(contract.class_names),
        }
        if labeled:
            arrays['labels'] = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        np.savez(folder / 'data.npz', **arrays)
        (folder / 'manifest.json').write_text(json.dumps({
            'schema_version': 1, 'dataset_name': 'fixed_test_fixture', 'partition_name': role,
            'role': role, 'client_id': None, 'data_format': 'npz', 'data_path': 'data.npz',
            'preprocessing_path': None, 'contract': contract.to_dict(),
            'testing_only': data_testing,
        }), encoding='utf-8')
        config = {
            'schema_version': 1, 'mode': mode,
            'experiment': {
                'name': 'prepared_test', 'seed': 42, 'device': 'cpu',
                'deterministic': deterministic, 'output_dir': '../runs',
            },
            'model': {'artifact_path': 'model.pt'},
            'dataset': {'manifest_path': 'manifest.json', 'expected_role': role},
            'inference': {'batch_size': 2},
        }
        path = folder / 'config.yaml'
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        return path

    return make


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _rewrite_yaml(path, change):
    """Change one fixture setting without bypassing the real config loader."""
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    change(settings)
    path.write_text(yaml.safe_dump(settings), encoding='utf-8')


def _forbid_seeding(monkeypatch):
    """Fail if invalid inputs reach the process-wide execution setup."""
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid inputs must be rejected before seeding')

    monkeypatch.setattr(runner, 'seed_everything', forbidden)


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_complete_cpu_run_outputs_known_answers(artifacts, mode):
    path = artifacts(mode)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    threads = torch.get_num_threads()
    directory = runner.run_prepared(path)

    assert isinstance(directory, Path) and directory.is_absolute()
    assert directory.parent == path.parent.parent / 'runs'
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    assert torch.get_num_threads() == threads
    summary = _read_json(directory / 'run_summary.json')
    assert summary['status'] == 'completed' and summary['sample_count'] == 5
    assert summary['mode'] == mode and summary['testing_only'] is True
    metadata = _read_json(directory / 'metadata.json')
    environment = metadata['environment']
    assert environment['seed'] == 42 and environment['device'] == 'cpu'
    assert environment['deterministic'] is True and environment['testing_only'] is True
    assert environment['config_path'] == str(path.resolve())
    assert environment['pytorch_threads'] == threads
    assert environment['execution_duration_seconds'] >= 0
    assert environment['git_commit'] == 'a' * 40
    assert 'testing-only verification' in environment['purpose']
    assert metadata['model']['training_completed'] is False
    expected_output = 'predictions.csv' if mode == 'predict' else 'final_metrics.json'
    assert {p.name for p in directory.iterdir()} == {
        'config.yaml', 'metadata.json', 'run_summary.json', expected_output,
    }
    if mode == 'predict':
        with (directory / expected_output).open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row['row_index']) for row in rows] == list(range(5))
        assert [int(row['class_id']) for row in rows] == [0, 1, 0, 1, 0]
        assert [row['class_name'] for row in rows] == [
            'class_0', 'class_1', 'class_0', 'class_1', 'class_0',
        ]
    else:
        metrics = _read_json(directory / expected_output)['metrics']
        # Four correct answers out of five are a software check, not an IDS result.
        assert metrics['accuracy'] == pytest.approx(0.8)
        assert metrics['confusion_matrix'] == [[2, 0], [1, 2]]
        assert [entry['class_name'] for entry in metrics['per_class']] == ['class_0', 'class_1']


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test', 'predict'])
def test_explicit_roles_are_retained_without_inference(artifacts, mode, role):
    directory = runner.run_prepared(artifacts(mode, role=role))
    metadata = _read_json(directory / 'metadata.json')
    assert metadata['dataset']['role'] == role
    assert _read_json(directory / 'run_summary.json')['testing_only'] is True


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
@pytest.mark.parametrize('model_testing,data_testing', [
    (False, False), (False, True), (True, False), (True, True),
])
def test_run_purpose_combines_model_and_data_declarations(
    artifacts, mode, model_testing, data_testing,
):
    directory = runner.run_prepared(artifacts(
        mode, model_testing=model_testing, data_testing=data_testing,
    ))
    expected = model_testing or data_testing
    metadata = _read_json(directory / 'metadata.json')
    assert metadata['testing_only'] is expected
    assert metadata['environment']['testing_only'] is expected
    assert ('testing-only verification' in metadata['environment']['purpose']) is expected
    assert _read_json(directory / 'run_summary.json')['testing_only'] is expected


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_repeated_runs_preserve_results_in_separate_folders(artifacts, mode):
    path = artifacts(mode)
    first = runner.run_prepared(path)
    random.random()
    np.random.rand()
    torch.rand(4)
    second = runner.run_prepared(path)
    assert first != second
    filename = 'predictions.csv' if mode == 'predict' else 'final_metrics.json'
    assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_feature_and_class_counts_come_from_artifacts(artifacts, mode):
    directory = runner.run_prepared(artifacts(mode, input_dim=4, num_classes=3))
    record = _read_json(directory / 'metadata.json')
    assert record['model']['contract']['input_dim'] == 4
    assert record['model']['contract']['num_classes'] == 3
    if mode == 'predict':
        with (directory / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            assert len(list(reader)) == 5
            assert 'probability_2' in reader.fieldnames
    else:
        matrix = _read_json(directory / 'final_metrics.json')['metrics']['confusion_matrix']
        assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)


def test_unavailable_cuda_fails_before_artifact_loading(artifacts, tmp_path, monkeypatch):
    path = artifacts()
    _rewrite_yaml(path, lambda config: config['experiment'].update(device='cuda'))

    def forbidden(*args, **kwargs):
        raise AssertionError('Artifacts must not load when requested CUDA is unavailable')

    monkeypatch.setattr(runner, 'load_model_artifact', forbidden)
    _forbid_seeding(monkeypatch)
    with pytest.raises(ValueError, match='CUDA.*unavailable'):
        runner.run_prepared(path)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('problem', [
    'invalid_config', 'missing_model', 'missing_manifest', 'missing_arrays',
    'model_schema', 'role', 'feature_contract', 'class_contract', 'labels', 'nan_features',
])
def test_invalid_inputs_fail_before_seeding_or_output_writes(
    artifacts, tmp_path, monkeypatch, problem,
):
    path = artifacts('evaluate', role='predict', labeled=problem != 'labels')
    if problem == 'invalid_config':
        _rewrite_yaml(path, lambda config: config['inference'].update(batch_size=0))
    elif problem == 'missing_model':
        (path.parent / 'model.pt').unlink()
    elif problem == 'missing_manifest':
        (path.parent / 'manifest.json').unlink()
    elif problem == 'missing_arrays':
        (path.parent / 'data.npz').unlink()
    elif problem == 'model_schema':
        torch.save({'model_state': {}}, path.parent / 'model.pt')
    elif problem == 'role':
        _rewrite_yaml(path, lambda config: config['dataset'].update(expected_role='test'))
    elif problem in ('feature_contract', 'class_contract'):
        manifest = _read_json(path.parent / 'manifest.json')
        field = 'feature_names' if problem == 'feature_contract' else 'class_names'
        manifest['contract'][field].reverse()
        (path.parent / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    elif problem == 'nan_features':
        with np.load(path.parent / 'data.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays['features'][0, 0] = float('nan')
        np.savez(path.parent / 'data.npz', **arrays)
    _forbid_seeding(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run_prepared(path)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_execution_failure_propagates_without_creating_logs(
    artifacts, tmp_path, monkeypatch, mode,
):
    path = artifacts(mode)

    def fail(*args, **kwargs):
        raise RuntimeError('simulated execution failure')

    name = 'predict_prepared_data' if mode == 'predict' else 'evaluate_prepared_data'
    monkeypatch.setattr(runner, name, fail)
    with pytest.raises(RuntimeError, match='simulated'):
        runner.run_prepared(path)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_no_optimizer_or_backpropagation_is_used(artifacts, monkeypatch, mode):
    path = artifacts(mode)

    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared execution must not optimize weights or backpropagate')

    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    assert (runner.run_prepared(path) / 'run_summary.json').exists()


def test_runner_passes_seed_and_deterministic_setting_to_shared_utility(
    artifacts, monkeypatch,
):
    path = artifacts(deterministic=False)
    _rewrite_yaml(path, lambda config: config['experiment'].update(seed=123))
    original = runner.seed_everything
    calls = []

    def observe(seed, deterministic):
        calls.append((seed, deterministic))
        original(seed, deterministic)

    monkeypatch.setattr(runner, 'seed_everything', observe)
    directory = runner.run_prepared(path)
    assert calls == [(123, False)]
    assert torch.are_deterministic_algorithms_enabled() is False
    metadata = _read_json(directory / 'metadata.json')['environment']
    assert metadata['seed'] == 123 and metadata['deterministic'] is False


def test_config_paths_do_not_depend_on_current_working_directory(
    artifacts, tmp_path, monkeypatch,
):
    path = artifacts()
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    directory = runner.run_prepared(path)
    assert directory.parent == tmp_path / 'runs'


def test_prediction_labels_do_not_change_predictions(artifacts):
    path = artifacts(labeled=True)
    first = runner.run_prepared(path)
    with np.load(path.parent / 'data.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays['labels'] = 1 - arrays['labels']
    np.savez(path.parent / 'data.npz', **arrays)
    second = runner.run_prepared(path)
    assert (first / 'predictions.csv').read_bytes() == (second / 'predictions.csv').read_bytes()
