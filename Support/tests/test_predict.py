"""Verify inference using fixed artificial weights; no model is trained."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from support.data.contract import ModelDataContract
from support.data.prepared import load_prepared_data
from support.evaluation.predict import predict_prepared_data
from support.models.factory import create_model
from support.utils.parameters import clone_state


@pytest.fixture
def prediction_fixture(tmp_path):
    description = {
        'schema_version': 1,
        'dataset_name': 'test_fixture',
        'partition_name': 'prediction',
        'role': 'predict',
        'client_id': None,
        'data_format': 'npz',
        'data_path': 'fixture.npz',
        'preprocessing_path': None,
        'testing_only': True,
        'contract': {
            'schema_version': 1,
            'model_type': 'mlp',
            'input_dim': 2,
            'num_classes': 2,
            'hidden_dims': [2],
            'feature_names': ['feature_a', 'feature_b'],
            'class_names': ['class_a', 'class_b'],
            'preprocessing_sha256': None,
        },
    }
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(description), encoding='utf-8')
    np.savez(
        tmp_path / 'fixture.npz',
        features=np.array([[3, 1], [1, 4], [2, 2], [0, 3], [5, 0]], dtype=np.float32),
        feature_names=np.array(['feature_a', 'feature_b']),
        class_names=np.array(['class_a', 'class_b']),
    )
    data = load_prepared_data(path, expected_role='predict')
    contract = ModelDataContract.from_dict(description['contract'])
    model = create_model(contract.model_config())
    # Hand-set weights make logits exactly 2 * features on these nonnegative
    # rows. This provides known answers without fitting on any dataset.
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(2))
        model.network[0].bias.zero_()
        model.network[2].weight.copy_(2 * torch.eye(2))
        model.network[2].bias.zero_()
    return model, data, contract


@pytest.mark.parametrize('batch_size', [1, 2, 5, 10])
def test_known_predictions_and_row_order(prediction_fixture, batch_size):
    model, data, contract = prediction_fixture

    result = predict_prepared_data(model, data, model_contract=contract, batch_size=batch_size)

    assert result.class_ids.tolist() == [0, 1, 0, 1, 0]
    assert result.predicted_labels == ('class_a', 'class_b', 'class_a', 'class_b', 'class_a')
    assert result.class_ids.dtype == torch.int64
    assert result.class_ids.device.type == 'cpu'
    assert result.probabilities.shape == (5, 2)
    assert result.probabilities.device.type == 'cpu'
    assert not result.probabilities.requires_grad
    assert torch.allclose(result.probabilities.sum(dim=1), torch.ones(5))
    assert torch.equal(result.probabilities[2], torch.tensor([0.5, 0.5]))
    assert result.probabilities[0, 0].item() == pytest.approx(1 / (1 + np.exp(-4)), rel=1e-6)
    assert result.manifest is data.manifest
    assert result.manifest.testing_only is True


def test_target_labels_cannot_change_predictions(prediction_fixture):
    model, data, contract = prediction_fixture
    features = data.dataset.tensors[0]
    first = replace(data, dataset=TensorDataset(features.clone(), torch.zeros(5, dtype=torch.int64)))
    second = replace(data, dataset=TensorDataset(features.clone(), torch.ones(5, dtype=torch.int64)))

    a = predict_prepared_data(model, first, model_contract=contract, batch_size=2)
    b = predict_prepared_data(model, second, model_contract=contract, batch_size=2)
    unlabeled = predict_prepared_data(model, data, model_contract=contract, batch_size=2)

    assert torch.equal(a.class_ids, b.class_ids)
    assert torch.equal(a.class_ids, unlabeled.class_ids)
    assert torch.equal(a.probabilities, b.probabilities)
    assert torch.equal(a.probabilities, unlabeled.probabilities)


def test_state_inputs_gradients_and_mixed_modes_are_preserved(prediction_fixture):
    model, data, contract = prediction_fixture
    model.train()
    model.network[1].eval()
    before = clone_state(model.state_dict())
    features = data.dataset.tensors[0].clone()
    modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]

    predict_prepared_data(model, data, model_contract=contract, batch_size=2)

    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)
    assert torch.equal(features, data.dataset.tensors[0])
    assert all(torch.equal(parameter.grad, gradient)
               for parameter, gradient in zip(model.parameters(), gradients))


def test_forward_uses_eval_and_inference_mode(prediction_fixture):
    model, data, contract = prediction_fixture
    observations = []

    def observe(module, args):
        observations.append((module.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

    model.register_forward_pre_hook(observe)

    predict_prepared_data(model, data, model_contract=contract, batch_size=2)

    assert observations == [(False, False, True)] * 3
    assert model.training is True


def test_eval_model_stays_in_eval(prediction_fixture):
    model, data, contract = prediction_fixture
    model.eval()

    predict_prepared_data(model, data, model_contract=contract)

    assert all(not module.training for module in model.modules())


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5, '2'])
def test_invalid_batch_sizes(prediction_fixture, batch_size):
    model, data, contract = prediction_fixture

    with pytest.raises(ValueError):
        predict_prepared_data(model, data, model_contract=contract, batch_size=batch_size)


@pytest.mark.parametrize('wrong_type', ['model', 'data', 'contract', 'dataset'])
def test_wrong_argument_types(prediction_fixture, wrong_type):
    model, data, contract = prediction_fixture
    if wrong_type == 'model':
        model = object()
    elif wrong_type == 'data':
        data = object()
    elif wrong_type == 'contract':
        contract = {}
    else:
        data = replace(data, dataset=[])

    with pytest.raises(TypeError):
        predict_prepared_data(model, data, model_contract=contract)


@pytest.mark.parametrize('field,value', [
    ('feature_names', ['feature_b', 'feature_a']),
    ('class_names', ['class_b', 'class_a']),
    ('hidden_dims', [3]),
])
def test_contract_mismatch_precedes_inference(prediction_fixture, field, value):
    model, data, contract = prediction_fixture
    description = contract.to_dict()
    description[field] = value
    expected = ModelDataContract.from_dict(description)

    def forbidden_forward(*args):
        raise AssertionError('The model must not run with an incompatible contract')

    model.register_forward_pre_hook(forbidden_forward)

    with pytest.raises(ValueError, match=field):
        predict_prepared_data(model, data, model_contract=expected)


@pytest.mark.parametrize('problem', [
    'integer', 'complex', 'nan', 'infinity', 'one_dimensional',
    'wrong_classes', 'wrong_rows', 'list', 'tuple', 'dictionary',
])
def test_invalid_outputs_restore_all_modes(prediction_fixture, problem):
    model, data, contract = prediction_fixture
    model.train()
    model.network[1].eval()
    before_modes = [module.training for module in model.modules()]

    def change_output(module, args, output):
        if problem == 'integer':
            return output.long()
        if problem == 'complex':
            return output.to(torch.complex64)
        if problem == 'nan':
            return torch.full_like(output, float('nan'))
        if problem == 'infinity':
            return torch.full_like(output, float('inf'))
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

    model.register_forward_hook(change_output)

    with pytest.raises(ValueError):
        predict_prepared_data(model, data, model_contract=contract)

    assert [module.training for module in model.modules()] == before_modes


def test_execution_error_restores_modes(prediction_fixture):
    model, data, contract = prediction_fixture
    model.train()
    model.network[1].eval()
    before_modes = [module.training for module in model.modules()]

    def fail_forward(*args):
        raise RuntimeError('test forward failure')

    model.register_forward_pre_hook(fail_forward)

    with pytest.raises(RuntimeError, match='test forward failure'):
        predict_prepared_data(model, data, model_contract=contract)

    assert [module.training for module in model.modules()] == before_modes


@pytest.mark.parametrize('problem', [
    'empty', 'extra_tensor', 'nan', 'integer', 'wrong_columns', 'three_dimensions',
])
def test_invalid_mutated_datasets(prediction_fixture, problem):
    model, data, contract = prediction_fixture
    features = data.dataset.tensors[0].clone()
    if problem == 'empty':
        dataset = TensorDataset(torch.zeros(0, 2))
    elif problem == 'extra_tensor':
        dataset = TensorDataset(features, features.clone(), features.clone())
    elif problem == 'nan':
        features[0, 0] = float('nan')
        dataset = TensorDataset(features)
    elif problem == 'integer':
        dataset = TensorDataset(features.long())
    elif problem == 'wrong_columns':
        dataset = TensorDataset(torch.zeros(5, 3))
    else:
        dataset = TensorDataset(features.unsqueeze(1))
    data = replace(data, dataset=dataset)

    with pytest.raises(ValueError):
        predict_prepared_data(model, data, model_contract=contract)

    assert model.training is True


def test_loader_does_not_consume_global_rng(prediction_fixture):
    model, data, contract = prediction_fixture
    before = torch.get_rng_state().clone()

    predict_prepared_data(model, data, model_contract=contract, batch_size=2)

    assert torch.equal(before, torch.get_rng_state())


def test_logits_decision_survives_softmax_rounding_tie(prediction_fixture):
    model, data, contract = prediction_fixture

    def tiny_difference(module, args, output):
        return torch.tensor([0., 1e-9], dtype=torch.float32).repeat(len(output), 1)

    model.register_forward_hook(tiny_difference)

    result = predict_prepared_data(model, data, model_contract=contract)

    assert torch.equal(result.probabilities, torch.full((5, 2), 0.5))
    assert result.class_ids.tolist() == [1] * 5


def test_unsupported_model_device(prediction_fixture):
    model, data, contract = prediction_fixture
    model.to('meta')

    with pytest.raises(ValueError, match='CPU or CUDA'):
        predict_prepared_data(model, data, model_contract=contract)


def test_mixed_model_devices(prediction_fixture):
    model, data, contract = prediction_fixture
    model.register_buffer('test_buffer', torch.zeros(1, device='meta'))

    with pytest.raises(ValueError, match='one device'):
        predict_prepared_data(model, data, model_contract=contract)
