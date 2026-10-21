"""Checkpoint verification with artificial weights only; no model training."""

from collections import OrderedDict
import pickle

import pytest
import torch

from support.models.checkpoint import load_model_checkpoint
from support.models.mlp import MLP
from support.utils.parameters import clone_state


def make_model():
    return MLP(input_dim=4, num_classes=3, hidden_dims=[5])


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('reverse_order', [False, True])
def test_loads_saved_weights_and_predictions(tmp_path, wrapped, reverse_order):
    source = make_model()
    # Fixed artificial weights isolate loading behavior from initialization RNG.
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index / 10)
    state = clone_state(source.state_dict())
    if reverse_order:
        state = OrderedDict(reversed(list(state.items())))
    saved = {'model_state': state, 'testing_only': True} if wrapped else state
    path = tmp_path / 'weights.pt'
    torch.save(saved, path)

    target = make_model().eval()
    returned = load_model_checkpoint(target, path)

    assert returned is target
    assert target.training is False
    assert next(target.parameters()).device.type == 'cpu'
    for name, expected in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], expected)
    with torch.no_grad():
        features = torch.zeros(2, 4)
        assert torch.equal(target(features), source(features))


@pytest.mark.parametrize('problem', [
    'missing_key', 'extra_key', 'wrong_shape', 'wrong_dtype', 'nan',
    'infinity', 'non_tensor', 'empty', 'invalid_wrapper', 'not_mapping',
])
def test_rejects_invalid_state_before_changing_model(tmp_path, problem):
    target = make_model()
    before = clone_state(target.state_dict())
    state = clone_state(make_model().state_dict())
    key = next(iter(state))
    if problem == 'missing_key':
        del state[key]
    elif problem == 'extra_key':
        state['unexpected'] = torch.zeros(1)
    elif problem == 'wrong_shape':
        state[key] = torch.zeros(1)
    elif problem == 'wrong_dtype':
        state[key] = state[key].double()
    elif problem == 'nan':
        state[key].fill_(float('nan'))
    elif problem == 'infinity':
        state[key].fill_(float('inf'))
    elif problem == 'non_tensor':
        state[key] = 1
    elif problem == 'empty':
        state = {}
    elif problem == 'invalid_wrapper':
        state = {'model_state': 1}
    elif problem == 'not_mapping':
        state = [1, 2]
    path = tmp_path / 'invalid.pt'
    torch.save(state, path)

    with pytest.raises(ValueError):
        load_model_checkpoint(target, path)

    assert target.training is True
    for name, expected in before.items():
        assert torch.equal(target.state_dict()[name], expected)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(make_model(), tmp_path / 'missing.pt')


def test_requires_pytorch_model(tmp_path):
    with pytest.raises(TypeError):
        load_model_checkpoint(object(), tmp_path / 'unused.pt')


def test_preserves_training_mode(tmp_path):
    model = make_model()
    path = tmp_path / 'weights.pt'
    torch.save(model.state_dict(), path)

    load_model_checkpoint(model, path)

    assert model.training is True


def test_corrupt_file_does_not_change_model(tmp_path):
    model = make_model()
    before = clone_state(model.state_dict())
    path = tmp_path / 'corrupt.pt'
    path.write_bytes(b'not a PyTorch checkpoint')

    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
        load_model_checkpoint(model, path)

    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected)
