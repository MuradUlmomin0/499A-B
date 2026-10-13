import pytest
import torch
from support.federated.client import FederatedClient
from support.models.factory import create_model
from support.data.dummy import make_dummy_data
from support.utils.parameters import clone_state


def test_client_training_and_evaluation(config):
    datasets,evaluation=make_dummy_data(6,3,[16],12,42)
    factory=lambda:create_model(config['model'])
    client=FederatedClient('a',factory,datasets[0],config['training'],evaluation_dataset=evaluation)
    with pytest.raises(ValueError):
        client.train(1)
    global_state=clone_state(factory().state_dict())
    client.receive_global(global_state)
    result=client.train(1)
    assert result.sample_count==16
    assert any(torch.count_nonzero(v) for v in result.update.values())
    for k in global_state:
        assert torch.allclose(result.update[k],result.state[k]-global_state[k])
    client.receive_global(global_state)
    repeat=client.train(1)
    assert all(torch.equal(result.state[k],repeat.state[k]) for k in result.state)
    before=clone_state(client.model.state_dict())
    assert 'macro_f1' in client.evaluate(3)
    assert all(torch.equal(before[k],client.model.state_dict()[k]) for k in before)
