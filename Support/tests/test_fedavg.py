import pytest
import torch
from support.federated.types import ClientResult
from support.federated.aggregation import FedAvgAggregator


def result(client,n,value):
    return ClientResult(client,1,n,{'w':torch.tensor([value],dtype=torch.float32)}, {}, {})


def test_weighted_average():
    a,b=result('a',1,2),result('b',3,10)
    state,weights=FedAvgAggregator().aggregate([a,b])
    assert state['w'].item()==8
    assert weights=={'a':0.25,'b':0.75}
    assert sum(weights.values())==pytest.approx(1)
    state['w'].zero_()
    assert b.state['w'].item()==10


@pytest.mark.parametrize('items',[[],[result('a',0,2)],[result('a',1,2),result('a',1,4)],[result('a',1,float('nan'))]])
def test_invalid(items):
    with pytest.raises(ValueError):
        FedAvgAggregator().aggregate(items)


def test_nonfloating_buffers_rejected():
    item=result('a',1,2)
    item.state={'counter':torch.tensor(2)}
    with pytest.raises(ValueError):
        FedAvgAggregator().aggregate([item])
