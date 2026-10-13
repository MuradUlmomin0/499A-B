import torch
import pytest
from support.data.dummy import make_dummy_data
from support.data.datasets import tabular_dataset


def test_dummy_independent_and_reproducible():
    a,e=make_dummy_data(4,3,[8,12],9,42)
    b,f=make_dummy_data(4,3,[8,12],9,42)
    assert [len(d) for d in a]==[8,12]
    assert torch.equal(e.tensors[0],f.tensors[0])
    assert torch.equal(a[0].tensors[0],b[0].tensors[0])
    assert a[0].tensors[0].data_ptr()!=e.tensors[0].data_ptr()
    rows=[tuple(row.tolist()) for d in a+[e] for row in d.tensors[0]]
    assert len(rows)==len(set(rows))


def test_invalid_labels():
    with pytest.raises(ValueError):
        tabular_dataset([[1,2]],[3],2,3)
