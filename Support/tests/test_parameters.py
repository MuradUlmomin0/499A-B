from collections import OrderedDict
import pytest
import torch
from support.utils.parameters import clone_state,extract_update,flatten_update,restore_update


def test_update_roundtrip_and_isolation():
    global_state=OrderedDict(a=torch.tensor([[1.,2.]]),b=torch.tensor([3.]))
    local=OrderedDict(a=torch.tensor([[4.,6.]]),b=torch.tensor([8.]))
    update=extract_update(local,global_state)
    assert torch.equal(update['a'],torch.tensor([[3.,4.]]))
    vector=flatten_update(update)
    assert vector.tolist()==[3.,4.,5.]
    restored=restore_update(vector,update)
    assert all(torch.equal(restored[k],update[k]) for k in update)
    copied=clone_state(global_state)
    copied['a'].zero_()
    restored['a'].zero_()
    assert global_state['a'].tolist()==[[1.,2.]]
    assert local['a'].tolist()==[[4.,6.]]
    assert update['a'].tolist()==[[3.,4.]]


def test_invalid_shape():
    with pytest.raises(ValueError):
        extract_update({'w':torch.ones(2)},{'w':torch.ones(3)})
    with pytest.raises(ValueError):
        restore_update(torch.ones(2),{'w':torch.ones(3)})
