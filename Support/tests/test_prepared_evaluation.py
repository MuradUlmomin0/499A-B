"""Prepared evaluation checks using fixed artificial weights; no training.

Synthetic arrays and model files are temporary, testing-only fixtures. Purpose
flag combinations exercise producer declarations; they do not establish actual
training, real-data provenance, or research results.
"""

from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from support.data.contract import ModelDataContract
from support.data.prepared import load_prepared_data
from support.evaluation.prepared import PreparedEvaluationResult, evaluate_prepared_data
from support.models.artifact import load_model_artifact
from support.models.factory import create_model
from support.utils.parameters import clone_state


@pytest.fixture
def evaluation_fixture(tmp_path):
    """Load a small artifact and labeled data through their normal public APIs."""
    contract = ModelDataContract(
        model_type='mlp', input_dim=2, num_classes=2, hidden_dims=(2,),
        feature_names=('feature_a', 'feature_b'),
        class_names=('class_a', 'class_b'), preprocessing_sha256=None,
    )
    model = create_model(contract.model_config())
    # These hand-set weights give logits 2 * features on nonnegative inputs.
    # Predictions are [0, 1, 0, 1, 0]; one differs from the supplied true label.
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(2))
        model.network[0].bias.zero_()
        model.network[2].weight.copy_(2 * torch.eye(2))
        model.network[2].bias.zero_()
    artifact_path = tmp_path / 'model.pt'
    torch.save({
        'schema_version': 1,
        'model_name': 'fixed_test_fixture',
        'model_state': clone_state(model.state_dict()),
        'contract': contract.to_dict(),
        'training_completed': False,
        'testing_only': True,
    }, artifact_path)
    artifact = load_model_artifact(artifact_path)

    np.savez(
        tmp_path / 'data.npz',
        features=np.array([[3, 1], [1, 4], [2, 2], [0, 3], [5, 0]], dtype=np.float32),
        labels=np.array([0, 1, 1, 1, 0], dtype=np.int64),
        feature_names=np.array(contract.feature_names),
        class_names=np.array(contract.class_names),
    )
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps({
        'schema_version': 1,
        'dataset_name': 'fixed_test_fixture',
        'partition_name': 'test_fixture',
        'role': 'test',
        'client_id': None,
        'data_format': 'npz',
        'data_path': 'data.npz',
        'preprocessing_path': None,
        'contract': contract.to_dict(),
        'testing_only': True,
    }), encoding='utf-8')
    data = load_prepared_data(
        manifest_path, expected_role='test', expected_contract=artifact.contract,
    )
    return artifact, data


@pytest.mark.parametrize('batch_size', [1, 2, 3, 5, 10])
def test_known_metrics_and_sample_weighted_loss(evaluation_fixture, batch_size):
    artifact, data = evaluation_fixture

    result = evaluate_prepared_data(artifact, data, expected_role='test', batch_size=batch_size)

    assert isinstance(result, PreparedEvaluationResult)
    assert result.sample_count == 5
    assert result.metrics['accuracy'] == pytest.approx(0.8)
    assert result.metrics['balanced_accuracy'] == pytest.approx(5 / 6)
    assert result.metrics['macro_precision'] == pytest.approx(5 / 6)
    assert result.metrics['macro_recall'] == pytest.approx(5 / 6)
    assert result.metrics['macro_f1'] == pytest.approx(0.8)
    assert result.metrics['weighted_f1'] == pytest.approx(0.8)
    assert result.metrics['confusion_matrix'] == [[2, 0], [1, 2]]
    per_class = result.metrics['per_class']
    assert [entry['class_name'] for entry in per_class] == ['class_a', 'class_b']
    assert [entry['support'] for entry in per_class] == [2, 3]
    assert [entry['precision'] for entry in per_class] == pytest.approx([2 / 3, 1])
    assert [entry['recall'] for entry in per_class] == pytest.approx([1, 2 / 3])
    assert [entry['f1'] for entry in per_class] == pytest.approx([0.8, 0.8])

    # Independently compute -log p(true class) from the known logits. Uneven
    # final batches must not receive the same weight as a full batch.
    features, labels = data.dataset.tensors
    logits = 2 * features.numpy().astype(np.float64)
    expected_loss = np.mean(
        np.logaddexp(logits[:, 0], logits[:, 1]) - logits[np.arange(5), labels.numpy()],
    )
    assert result.metrics['loss'] == pytest.approx(expected_loss, rel=1e-12, abs=1e-14)
    json.dumps(result.metrics, allow_nan=False)


def test_source_metadata_is_preserved(evaluation_fixture):
    artifact, data = evaluation_fixture

    result = evaluate_prepared_data(artifact, data, expected_role='test')

    assert result.manifest is data.manifest
    assert result.model_name == artifact.model_name
    assert result.artifact_path == artifact.artifact_path
    assert result.artifact_sha256 == artifact.artifact_sha256
    assert result.training_completed is False
    assert result.testing_only is True


@pytest.mark.parametrize('model_testing,data_testing', [
    (False, False), (False, True), (True, False), (True, True),
])
def test_testing_only_status_combines_both_inputs(evaluation_fixture, model_testing, data_testing):
    # These are metadata cases, including claimed training completion. No model
    # is actually trained and a false testing-only flag cannot prove provenance.
    artifact, data = evaluation_fixture
    artifact = replace(artifact, training_completed=True, testing_only=model_testing)
    data = replace(data, manifest=replace(data.manifest, testing_only=data_testing))

    result = evaluate_prepared_data(artifact, data, expected_role='test')

    assert result.testing_only is (model_testing or data_testing)
    assert result.training_completed is True


@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test', 'predict'])
def test_explicit_labeled_roles_are_retained(evaluation_fixture, role):
    artifact, data = evaluation_fixture
    data = replace(data, manifest=replace(data.manifest, role=role))

    result = evaluate_prepared_data(artifact, data, expected_role=role)

    assert result.manifest.role == role


def test_role_mismatch_precedes_model_execution(evaluation_fixture):
    artifact, data = evaluation_fixture

    def forbidden_forward(*args):
        raise AssertionError('Role mismatch must prevent model execution')

    artifact.model.register_forward_pre_hook(forbidden_forward)

    with pytest.raises(ValueError, match='Expected role'):
        evaluate_prepared_data(artifact, data, expected_role='validation')


@pytest.mark.parametrize('role', ['', None, True, 'evaluation'])
def test_rejects_unsupported_expected_roles(evaluation_fixture, role):
    with pytest.raises(ValueError, match='supported data role'):
        evaluate_prepared_data(*evaluation_fixture, expected_role=role)


def test_requires_an_explicit_role(evaluation_fixture):
    with pytest.raises(TypeError, match='expected_role'):
        evaluate_prepared_data(*evaluation_fixture)


@pytest.mark.parametrize('batch_size', [0, -1, True, 2.5, '2'])
def test_rejects_invalid_batch_sizes(evaluation_fixture, batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        evaluate_prepared_data(*evaluation_fixture, expected_role='test', batch_size=batch_size)


@pytest.mark.parametrize('wrong_type', ['artifact', 'model', 'contract', 'data', 'dataset', 'manifest'])
def test_rejects_wrong_input_types(evaluation_fixture, wrong_type):
    artifact, data = evaluation_fixture
    if wrong_type == 'artifact':
        artifact = object()
    elif wrong_type == 'model':
        artifact = replace(artifact, model=object())
    elif wrong_type == 'contract':
        artifact = replace(artifact, contract={})
    elif wrong_type == 'data':
        data = object()
    elif wrong_type == 'dataset':
        data = replace(data, dataset=[])
    else:
        data = replace(data, manifest={})

    with pytest.raises(TypeError):
        evaluate_prepared_data(artifact, data, expected_role='test')


@pytest.mark.parametrize('field', ['training_completed', 'model_testing', 'data_testing'])
def test_rejects_invalid_purpose_flags(evaluation_fixture, field):
    artifact, data = evaluation_fixture
    if field == 'training_completed':
        artifact = replace(artifact, training_completed=1)
    elif field == 'model_testing':
        artifact = replace(artifact, testing_only='true')
    else:
        data = replace(data, manifest=replace(data.manifest, testing_only=1))

    with pytest.raises(ValueError, match='booleans'):
        evaluate_prepared_data(artifact, data, expected_role='test')


def test_declared_untrained_model_requires_testing_only(evaluation_fixture):
    artifact, data = evaluation_fixture
    artifact = replace(artifact, testing_only=False)

    with pytest.raises(ValueError, match='untrained'):
        evaluate_prepared_data(artifact, data, expected_role='test')


@pytest.mark.parametrize('field,value', [
    ('feature_names', ['feature_b', 'feature_a']),
    ('class_names', ['class_b', 'class_a']),
    ('hidden_dims', [3]),
    ('preprocessing_sha256', 'a' * 64),
])
def test_contract_mismatch_precedes_model_execution(evaluation_fixture, field, value):
    artifact, data = evaluation_fixture
    description = data.manifest.contract.to_dict()
    description[field] = value
    incompatible = ModelDataContract.from_dict(description)
    data = replace(data, manifest=replace(data.manifest, contract=incompatible))

    def forbidden_forward(*args):
        raise AssertionError('Contract mismatch must prevent model execution')

    artifact.model.register_forward_pre_hook(forbidden_forward)

    with pytest.raises(ValueError, match=field):
        evaluate_prepared_data(artifact, data, expected_role='test')


@pytest.mark.parametrize('problem', [
    'missing', 'empty', 'extra', 'no_tensors', 'float', 'int32', 'bool',
    'negative', 'out_of_range', 'two_dimensions',
])
def test_rejects_invalid_or_missing_labels(evaluation_fixture, problem):
    artifact, data = evaluation_fixture
    features, labels = (tensor.clone() for tensor in data.dataset.tensors)
    if problem == 'missing':
        dataset = TensorDataset(features)
    elif problem == 'empty':
        dataset = TensorDataset(features[:0], labels[:0])
    elif problem == 'extra':
        dataset = TensorDataset(features, labels, labels.clone())
    elif problem == 'no_tensors':
        dataset = TensorDataset()
    else:
        if problem == 'float':
            labels = labels.float()
        elif problem == 'int32':
            labels = labels.int()
        elif problem == 'bool':
            labels = labels.bool()
        elif problem == 'negative':
            labels[0] = -1
        elif problem == 'out_of_range':
            labels[0] = 2
        else:
            labels = labels.unsqueeze(1)
        dataset = TensorDataset(features, labels)
    data = replace(data, dataset=dataset)

    with pytest.raises(ValueError):
        evaluate_prepared_data(artifact, data, expected_role='test')


@pytest.mark.parametrize('problem', ['nan', 'inf', 'integer', 'wrong_columns', 'three_dimensions'])
def test_invalid_features_restore_modes(evaluation_fixture, problem):
    artifact, data = evaluation_fixture
    features, labels = (tensor.clone() for tensor in data.dataset.tensors)
    if problem == 'nan':
        features[-1, 0] = float('nan')
    elif problem == 'inf':
        features[-1, 0] = float('inf')
    elif problem == 'integer':
        features = features.long()
    elif problem == 'wrong_columns':
        features = torch.zeros(5, 3)
    else:
        features = features.unsqueeze(1)
    data = replace(data, dataset=TensorDataset(features, labels))
    artifact.model.train()
    artifact.model.network[1].eval()
    before_modes = [module.training for module in artifact.model.modules()]

    # The last-row failures also exercise cleanup after earlier valid batches.
    with pytest.raises(ValueError, match='features'):
        evaluate_prepared_data(artifact, data, expected_role='test', batch_size=2)

    assert [module.training for module in artifact.model.modules()] == before_modes


def test_weights_gradients_data_modes_and_rng_are_preserved(evaluation_fixture):
    artifact, data = evaluation_fixture
    model = artifact.model
    model.train()
    model.network[1].eval()
    before_modes = [module.training for module in model.modules()]
    before_state = clone_state(model.state_dict())
    before_data = [tensor.clone() for tensor in data.dataset.tensors]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    before_rng = torch.get_rng_state().clone()

    evaluate_prepared_data(artifact, data, expected_role='test', batch_size=2)

    assert [module.training for module in model.modules()] == before_modes
    assert all(torch.equal(value, before_state[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(before, after) for before, after in zip(before_data, data.dataset.tensors))
    assert all(torch.equal(parameter.grad, grad) for parameter, grad in zip(model.parameters(), gradients))
    assert torch.equal(before_rng, torch.get_rng_state())


def test_execution_uses_eval_and_inference_mode(evaluation_fixture):
    artifact, data = evaluation_fixture
    observations = []

    def observe(module, args):
        observations.append((module.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

    artifact.model.register_forward_pre_hook(observe)

    evaluate_prepared_data(artifact, data, expected_role='test', batch_size=2)

    assert observations == [(False, False, True)] * 3


@pytest.mark.parametrize('problem', [
    'integer', 'complex', 'nan', 'inf', 'one_dimensional',
    'wrong_classes', 'wrong_rows', 'list', 'tuple', 'dictionary',
])
def test_invalid_logits_restore_modes(evaluation_fixture, problem):
    artifact, data = evaluation_fixture
    model = artifact.model
    model.train()
    model.network[1].eval()
    before_modes = [module.training for module in model.modules()]

    def invalid_output(module, args, output):
        if problem == 'integer':
            return output.long()
        if problem == 'complex':
            return output.to(torch.complex64)
        if problem in ('nan', 'inf'):
            return torch.full_like(output, float(problem))
        if problem == 'one_dimensional':
            return output[:, 0]
        if problem == 'wrong_classes':
            return output[:, :1]
        if problem == 'wrong_rows':
            return output[:-1]
        if problem == 'list':
            return output.tolist()
        if problem == 'tuple':
            return (output,)
        return {'logits': output}

    model.register_forward_hook(invalid_output)

    with pytest.raises(ValueError):
        evaluate_prepared_data(artifact, data, expected_role='test')

    assert [module.training for module in model.modules()] == before_modes


def test_forward_failure_restores_modes(evaluation_fixture):
    artifact, data = evaluation_fixture
    artifact.model.train()
    artifact.model.network[1].eval()
    before_modes = [module.training for module in artifact.model.modules()]

    def fail_forward(*args):
        raise RuntimeError('intentional failure')

    artifact.model.register_forward_pre_hook(fail_forward)

    with pytest.raises(RuntimeError, match='intentional failure'):
        evaluate_prepared_data(artifact, data, expected_role='test')

    assert [module.training for module in artifact.model.modules()] == before_modes


def test_logits_argmax_retains_tiny_decision_difference(evaluation_fixture):
    artifact, data = evaluation_fixture
    data = replace(data, dataset=TensorDataset(data.dataset.tensors[0], torch.ones(5, dtype=torch.long)))

    def tiny_output(module, args, output):
        return torch.tensor([0., 1e-9]).repeat(len(output), 1)

    artifact.model.register_forward_hook(tiny_output)

    result = evaluate_prepared_data(artifact, data, expected_role='test')

    # Softmax would round these scores to a tie in float32. Evaluation must
    # retain the logits' class decision and the existing missing-class policy.
    assert result.metrics['accuracy'] == 1.0
    assert result.metrics['confusion_matrix'] == [[0, 0], [0, 5]]
    assert result.metrics['per_class'][0]['support'] == 0
    assert result.metrics['macro_recall'] == 0.5
    assert result.metrics['balanced_accuracy'] == 1.0


def test_float32_extreme_logits_have_finite_loss(evaluation_fixture):
    artifact, data = evaluation_fixture
    data = replace(data, dataset=TensorDataset(data.dataset.tensors[0], torch.ones(5, dtype=torch.long)))

    def extreme_output(module, args, output):
        return torch.tensor([3e38, -3e38], dtype=torch.float32).repeat(len(output), 1)

    artifact.model.register_forward_hook(extreme_output)

    result = evaluate_prepared_data(artifact, data, expected_role='test')

    expected_loss = 2 * torch.tensor(3e38, dtype=torch.float32).item()
    assert math.isfinite(result.metrics['loss'])
    assert result.metrics['loss'] == pytest.approx(expected_loss, rel=1e-12)


@pytest.mark.parametrize('problem', ['batch_loss', 'accumulation'])
def test_nonfinite_loss_fails_and_restores_modes(evaluation_fixture, problem):
    artifact, data = evaluation_fixture
    data = replace(data, dataset=TensorDataset(data.dataset.tensors[0], torch.ones(5, dtype=torch.long)))
    artifact.model.train()
    before_modes = [module.training for module in artifact.model.modules()]

    def extreme_output(module, args, output):
        values = [1e308, -1e308] if problem == 'batch_loss' else [0., -1e308]
        return torch.tensor(values, dtype=torch.float64).repeat(len(output), 1)

    artifact.model.register_forward_hook(extreme_output)

    with pytest.raises(ValueError, match='nonfinite'):
        evaluate_prepared_data(artifact, data, expected_role='test', batch_size=1)

    assert [module.training for module in artifact.model.modules()] == before_modes


@pytest.mark.parametrize('problem', ['unsupported', 'mixed'])
def test_rejects_invalid_model_devices(evaluation_fixture, problem):
    artifact, data = evaluation_fixture
    if problem == 'unsupported':
        artifact.model.to('meta')
    else:
        artifact.model.register_buffer('test_buffer', torch.zeros(1, device='meta'))

    with pytest.raises(ValueError):
        evaluate_prepared_data(artifact, data, expected_role='test')
