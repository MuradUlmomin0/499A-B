import random
import numpy as np
import torch
from support.utils.reproducibility import seed_everything


def test_seed_streams():
    seed_everything(42)
    a=(random.random(),np.random.rand(),torch.rand(3))
    seed_everything(42)
    b=(random.random(),np.random.rand(),torch.rand(3))
    assert a[:2]==b[:2]
    assert torch.equal(a[2],b[2])
