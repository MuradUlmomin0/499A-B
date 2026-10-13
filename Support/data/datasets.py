"""No preprocessing or dataset-specific feature assumptions."""
import torch
from torch.utils.data import TensorDataset


def tabular_dataset(features, labels, input_dim, num_classes):
    x = torch.as_tensor(features, dtype=torch.float32).detach().clone()
    y = torch.as_tensor(labels)
    if x.ndim != 2 or x.shape[1] != input_dim or x.shape[0] == 0 or not torch.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    if y.ndim != 1 or len(y) != len(x) or y.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Labels must be one integer per sample')
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError('Label outside configured classes')
    return TensorDataset(x, y.to(torch.long).detach().clone())
