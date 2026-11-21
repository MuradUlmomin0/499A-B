"""Supplied notebook's neural ensemble (cells 24–26), for inference only.

Names deliberately match the checkpoint. The notebook calls its components
logistic regression and extra trees; these are neural approximations, not the
scikit-learn algorithms. This is the end-to-end ensemble, not its separate
holdout-blending experiment. Do not pass its integer buffers to current FedAvg.
"""
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, num_classes, max_depth=10):
        super().__init__()
        self.max_depth = max_depth
        self.num_leaves = 2 ** max_depth
        self.routing = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                     nn.Linear(128, self.num_leaves), nn.Softmax(dim=-1))
        self.leaf_classifiers = nn.Parameter(torch.randn(self.num_leaves, num_classes))

    def forward(self, x):
        return torch.matmul(self.routing(x), self.leaf_classifiers)


class UpgradedLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.network(x)


class RandomTreeEstimatorApproximation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, 64)
        with torch.no_grad():
            self.projection.weight.normal_(0.0, 1.0)
            self.projection.bias.uniform_(-1.0, 1.0)
        for parameter in self.projection.parameters():
            parameter.requires_grad = False
        self.network = nn.Sequential(nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        return self.network(self.projection(x))


class DeepExtraTreesApproximation(nn.Module):
    def __init__(self, input_dim, num_classes, num_estimators=10):
        super().__init__()
        self.estimators = nn.ModuleList([
            RandomTreeEstimatorApproximation(input_dim, num_classes)
            for _ in range(num_estimators)])

    def forward(self, x):
        return torch.mean(torch.stack([estimator(x) for estimator in self.estimators]), dim=0)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.fc(x)))) + x


class UpgradedTabularClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.transition = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
                                        nn.ReLU(), nn.Dropout(0.3))
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.output_layer(self.transition(self.res_block2(self.res_block1(self.input_layer(x)))))


class PyTorchStackingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        if (type(input_dim) is not int or not 1 <= input_dim <= 512
                or type(num_classes) is not int or not 2 <= num_classes <= 256):
            raise ValueError('Unsupported stacking input/class dimensions')
        self.input_dim, self.num_classes = input_dim, num_classes
        self.base_model1 = SoftDecisionTree(input_dim, num_classes)
        self.base_model2 = UpgradedLogisticRegression(input_dim, num_classes)
        self.base_model3 = DeepExtraTreesApproximation(input_dim, num_classes)
        self.meta_model = UpgradedTabularClassifier(3 * num_classes, num_classes)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('Expected [batch, input_dim] stacking features')
        return self.meta_model(torch.cat([
            self.base_model1(x), self.base_model2(x), self.base_model3(x)], dim=1))


def load_stacking_checkpoint(path, input_dim, num_classes):
    """Strict CPU weights-only load; never infer feature/class semantics.

    Unlike the MLP/FedAvg state API, this accepts the notebook's nonnegative
    int64 batch counters. RNG is preserved and frozen projections stay frozen.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('Stacking checkpoint exceeds the supported 64 MiB limit')
    state = torch.load(path, map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        model = PyTorchStackingClassifier(input_dim, num_classes)
    reference = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(reference):
        raise ValueError('Stacking checkpoint parameter names do not match notebook architecture')
    for name, expected in reference.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid stacking checkpoint tensor: {name}')
        if name.endswith('num_batches_tracked') and value.item() < 0:
            raise ValueError(f'Invalid batch counter: {name}')
        if name.endswith('running_var') and (value < 0).any():
            raise ValueError(f'Invalid batch variance: {name}')
    model.load_state_dict(state, strict=True)
    return model.eval()
