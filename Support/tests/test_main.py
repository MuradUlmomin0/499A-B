"""Verify command-line routing and prepared execution without model training.

Prepared runs use temporary synthetic arrays and fixed artificial weights. Their
metrics check software behavior only. Dummy FedAvg routing uses a stand-in
function so these tests never start local training or a federated experiment.
"""

import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

import main as cli
from support.data.contract import ModelDataContract
from support.models.factory import create_model
from support.utils.parameters import clone_state
import support.utils.reproducibility as reproducibility


def _forbidden(*args, **kwargs):
    raise AssertionError('Unexpected workflow or training call')


@pytest.fixture(autouse=True)
def no_training_and_restore_state(monkeypatch):
    """Guard against training and restore settings changed by in-process runs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', _forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', _forbidden)
    monkeypatch.setattr(cli, 'run_dummy_test', _forbidden)

    def git_query(command, **kwargs):
        if command[-2:] == ['rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        if command[-2:] == ['status', '--porcelain']:
            return ''
        raise AssertionError('Unexpected subprocess request')

    # In-process metadata must not depend on the developer's Git history.
    # Separately launched scripts read their own environment and Git metadata.
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
def prepared_inputs(tmp_path):
    """Return a builder for complete testing-only prepared-run input files."""
    def make(mode='predict'):
        folder = tmp_path / mode
        folder.mkdir()
        contract = ModelDataContract(
            model_type='mlp', input_dim=2, num_classes=2, hidden_dims=(2,),
            feature_names=('feature_a', 'feature_b'), class_names=('class_a', 'class_b'),
            preprocessing_sha256=None,
        )
        model = create_model(contract.model_config())
        # Assign known weights directly; this fixture is never fitted.
        with torch.no_grad():
            model.network[0].weight.copy_(torch.eye(2))
            model.network[0].bias.zero_()
            model.network[2].weight.copy_(2 * torch.eye(2))
            model.network[2].bias.zero_()
        torch.save({
            'schema_version': 1, 'model_name': 'fixed_cli_test_fixture',
            'model_state': clone_state(model.state_dict()), 'contract': contract.to_dict(),
            'training_completed': False, 'testing_only': True,
        }, folder / 'model.pt')
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
            'schema_version': 1, 'dataset_name': 'synthetic_cli_fixture', 'partition_name': role,
            'role': role, 'client_id': None, 'data_format': 'npz', 'data_path': 'data.npz',
            'preprocessing_path': None, 'contract': contract.to_dict(), 'testing_only': True,
        }), encoding='utf-8')
        path = folder / 'config.yaml'
        path.write_text(yaml.safe_dump({
            'schema_version': 1, 'mode': mode,
            'experiment': {
                'name': 'prepared_cli_test', 'seed': 42, 'device': 'cpu',
                'deterministic': True, 'output_dir': '../runs',
            },
            'model': {'artifact_path': 'model.pt'},
            'dataset': {'manifest_path': 'manifest.json', 'expected_role': role},
            'inference': {'batch_size': 2},
        }), encoding='utf-8')
        return path

    return make


def _invoke(monkeypatch, arguments):
    """Simulate command arguments while calling the real main() function."""
    monkeypatch.setattr(sys, 'argv', ['main.py', *map(str, arguments)])
    return cli.main()


@pytest.mark.parametrize('arguments,message', [
    ([], 'required'),
    (['--config', 'ignored.yaml'], 'required'),
    (['--prepared'], 'requires --config'),
    (['--prepared', '--dummy-test'], 'not allowed with argument'),
    (['--dummy-test', '--prepared', '--config', 'ignored.yaml'], 'not allowed with argument'),
    (['--dummy-test', '--config'], 'expected one argument'),
    (['--prepared', '--config'], 'expected one argument'),
    (['--prepared', '--config', 'ignored.yaml', '--unknown'], 'unrecognized arguments'),
])
def test_invalid_arguments_do_not_load_or_execute(monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    monkeypatch.setattr(cli, 'run_prepared', _forbidden)
    with pytest.raises(SystemExit) as caught:
        _invoke(monkeypatch, arguments)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert message in captured.err


def test_help_explains_both_workflows_without_execution(monkeypatch, capsys):
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    monkeypatch.setattr(cli, 'run_prepared', _forbidden)
    with pytest.raises(SystemExit) as caught:
        _invoke(monkeypatch, ['--help'])
    assert caught.value.code == 0
    output = ' '.join(capsys.readouterr().out.split())
    for text in ('--dummy-test', '--prepared', '--config', 'without training'):
        assert text in output


@pytest.mark.parametrize('custom_config', [False, True])
def test_dummy_workflow_keeps_default_and_explicit_configs(
    monkeypatch, tmp_path, capsys, custom_config,
):
    default_path = cli.REPOSITORY / 'support/configs/fedavg_dummy_test.yaml'
    expected_path = default_path
    if custom_config:
        expected_path = tmp_path / 'custom dummy settings.yaml'
        expected_path.write_text(default_path.read_text(encoding='utf-8'), encoding='utf-8')
    expected = yaml.safe_load(expected_path.read_text(encoding='utf-8'))
    original_loader = cli.load_config
    loaded, executed = [], []

    def load(path):
        loaded.append(path)
        return original_loader(path)

    def dummy(config):
        executed.append(config)
        return tmp_path / 'dummy_artifacts', {}

    monkeypatch.setattr(cli, 'load_config', load)
    monkeypatch.setattr(cli, 'run_dummy_test', dummy)
    monkeypatch.setattr(cli, 'run_prepared', _forbidden)
    # The bundled default must still be found when launched outside the repo.
    monkeypatch.chdir(tmp_path)
    arguments = ['--dummy-test']
    if custom_config:
        arguments += ['--config', expected_path]
    assert _invoke(monkeypatch, arguments) is None
    assert loaded == [expected_path]
    assert executed == [expected]
    output = capsys.readouterr().out
    assert 'Synthetic FedAvg framework test completed. Metrics are NOT research results.' in output
    assert f'Test artifacts: {tmp_path / "dummy_artifacts"}' in output


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_prepared_routes_config_path_directly_to_runner(monkeypatch, tmp_path, capsys, mode):
    config_path = tmp_path / f'prepared {mode}.yaml'
    # This routing-only case replaces the runner; full configurations are used
    # by the end-to-end cases below. YAML validation belongs to the real runner.
    config_path.write_text(yaml.safe_dump({'mode': mode}), encoding='utf-8')
    calls = []

    def prepared(path):
        calls.append(path)
        return tmp_path / 'prepared_artifacts'

    monkeypatch.setattr(cli, 'run_prepared', prepared)
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    assert _invoke(monkeypatch, ['--prepared', '--config', config_path]) is None
    assert calls == [config_path]
    assert capsys.readouterr().out.splitlines() == [
        'Prepared-data run completed.', f'Run artifacts: {tmp_path / "prepared_artifacts"}',
    ]


@pytest.mark.parametrize('error', [
    FileNotFoundError('missing artifact'), ValueError('invalid contract'),
])
def test_prepared_failures_propagate_without_success_message(
    monkeypatch, tmp_path, capsys, error,
):
    def fail(path):
        raise error

    monkeypatch.setattr(cli, 'run_prepared', fail)
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    with pytest.raises(type(error), match=str(error)):
        _invoke(monkeypatch, ['--prepared', '--config', tmp_path / 'config.yaml'])
    assert not capsys.readouterr().out


@pytest.mark.parametrize('workflow', ['--dummy-test', '--prepared'])
def test_missing_config_fails_without_training_or_outputs(
    monkeypatch, tmp_path, capsys, workflow,
):
    with pytest.raises(FileNotFoundError):
        _invoke(monkeypatch, [workflow, '--config', tmp_path / 'missing.yaml'])
    assert not capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('workflow', ['--dummy-test', '--prepared'])
def test_wrong_schema_is_rejected_instead_of_switching_workflows(
    monkeypatch, prepared_inputs, tmp_path, capsys, workflow,
):
    if workflow == '--prepared':
        config_path = cli.REPOSITORY / 'support/configs/fedavg_dummy_test.yaml'
    else:
        config_path = prepared_inputs()
    with pytest.raises(ValueError):
        _invoke(monkeypatch, [workflow, '--config', config_path])
    assert not capsys.readouterr().out
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_prepared_end_to_end_uses_no_optimizer_or_backpropagation(
    monkeypatch, prepared_inputs, capsys, mode,
):
    path = prepared_inputs(mode)
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    assert _invoke(monkeypatch, ['--prepared', '--config', path]) is None
    output = capsys.readouterr().out.splitlines()
    assert output[0] == 'Prepared-data run completed.'
    directory = Path(output[1].removeprefix('Run artifacts: '))
    summary = json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed'
    assert summary['mode'] == mode
    assert summary['testing_only'] is True


@pytest.mark.parametrize('mode', ['predict', 'evaluate'])
def test_actual_script_executes_prepared_flow_from_another_directory(
    prepared_inputs, tmp_path, mode,
):
    path = prepared_inputs(mode)
    before = {item.name: item.read_bytes() for item in path.parent.iterdir()}
    working = tmp_path / 'unrelated working directory'
    working.mkdir()
    command = [
        sys.executable, str(cli.REPOSITORY / 'main.py'), '--prepared', '--config',
        os.path.relpath(path, working),
    ]
    completed = subprocess.run(command, cwd=working, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == 'Prepared-data run completed.'
    assert len(lines) == 2 and lines[1].startswith('Run artifacts: ')
    directory = Path(lines[1].removeprefix('Run artifacts: '))
    assert directory.parent == tmp_path / 'runs'
    summary = json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))
    assert summary['mode'] == mode and summary['testing_only'] is True
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['environment']['device'] == 'cpu'
    assert metadata['environment']['config_path'] == str(path.resolve())
    assert metadata['model']['training_completed'] is False
    assert before == {item.name: item.read_bytes() for item in path.parent.iterdir()}
    if mode == 'predict':
        with (directory / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row['class_id']) for row in rows] == [0, 1, 0, 1, 0]
    else:
        record = json.loads((directory / 'final_metrics.json').read_text(encoding='utf-8'))
        # The known toy labels produce four correct answers out of five.
        assert record['metrics']['accuracy'] == pytest.approx(0.8)
        assert record['metrics']['confusion_matrix'] == [[2, 0], [1, 2]]


@pytest.mark.parametrize('arguments', [
    ['--federated'], ['--federated', '--prepared'], ['--federated', '--dummy-test'],
])
def test_federated_arguments_are_explicit_and_exclusive(monkeypatch, capsys, arguments):
    monkeypatch.setattr(cli, 'run_federated', _forbidden)
    with pytest.raises(SystemExit) as caught:
        _invoke(monkeypatch, arguments)
    assert caught.value.code == 2
    assert not capsys.readouterr().out


def test_federated_routes_only_to_training_runner(monkeypatch, tmp_path, capsys):
    calls = []
    def run(path):
        calls.append(path)
        return tmp_path / 'output'
    monkeypatch.setattr(cli, 'run_federated', run)
    monkeypatch.setattr(cli, 'load_config', _forbidden)
    monkeypatch.setattr(cli, 'run_prepared', _forbidden)
    path = tmp_path / 'explicit.yaml'
    _invoke(monkeypatch, ['--federated', '--config', path])
    assert calls == [path]
    assert capsys.readouterr().out.splitlines() == [
        'Prepared FedAvg run completed (round evaluation: validation).', f'Run artifacts: {tmp_path / "output"}',
    ]


def test_federated_failure_prints_no_success(monkeypatch, tmp_path, capsys):
    def fail(path): raise ValueError('invalid bundle')
    monkeypatch.setattr(cli, 'run_federated', fail)
    with pytest.raises(ValueError, match='invalid bundle'):
        _invoke(monkeypatch, ['--federated', '--config', tmp_path / 'bad.yaml'])
    assert not capsys.readouterr().out


def test_federated_script_with_controlled_updates_and_no_optimizer(federated_config, tmp_path):
    path, _ = federated_config()
    working = tmp_path / 'external_launch'
    working.mkdir()
    script = '''
import sys, runpy
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
import torch
from support.federated.client import FederatedClient
from support.federated.types import ClientResult
from support.utils.parameters import clone_state, extract_update
def forbidden(*args, **kwargs):
    raise AssertionError('No optimizer/backpropagation allowed in CLI infrastructure check')
torch.optim.Optimizer.__init__ = forbidden
torch.Tensor.backward = forbidden
torch.cuda.is_available = lambda: False
torch.cuda.manual_seed_all = lambda seed: None
def controlled(self, round_id):
    with torch.no_grad():
        for parameter in self.model.parameters():
            parameter.add_(1 if self.client_id == 'a' else 3)
    state = clone_state(self.model.state_dict())
    return ClientResult(self.client_id, round_id, len(self.train_dataset), state,
                        extract_update(state, self.global_state), {'train_loss': 1.0})
FederatedClient.train = controlled
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script, str(cli.REPOSITORY / 'main.py'),
                             '--federated', '--config', os.path.relpath(path, working)],
                            cwd=working, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    directory = Path(result.stdout.splitlines()[1].removeprefix('Run artifacts: '))
    summary = json.loads((directory / 'run_summary.json').read_text())
    assert summary['status'] == 'completed' and summary['testing_only'] is True
    assert summary['selection_policy'] == 'final_round' and summary['selected_round'] == 2
