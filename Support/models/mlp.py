import torch
from torch import nn


class MLP(nn.Module):
    """Tabular classifier: [batch, input_dim] to unnormalized class logits."""
    def __init__(self, input_dim, num_classes, hidden_dims):
        super().__init__()
        if type(input_dim) is not int or input_dim < 1 or type(num_classes) is not int or num_classes < 2:
            raise ValueError('Invalid input or class dimension')
        if not hidden_dims or any(type(n) is not int or n < 1 for n in hidden_dims):
            raise ValueError('Invalid hidden dimensions')
        self.input_dim, self.num_classes = input_dim, num_classes
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('Expected [batch, input_dim] features')
        return self.network(x)
