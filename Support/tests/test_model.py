import pytest
import torch
from support.models.mlp import MLP


@pytest.mark.parametrize('features,classes',[(2,2),(9,5)])
def test_configurable(features,classes):
    model=MLP(features,classes,[7])
    logits=model(torch.randn(4,features))
    assert logits.shape==(4,classes)
    torch.nn.functional.cross_entropy(logits,torch.arange(4)%classes).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_bad_shape():
    with pytest.raises(ValueError):
        MLP(2,3,[4])(torch.zeros(2,3))
