import torch
from torch import nn


class LeNet(nn.Module):
    """
    Simple LeNet model for MNIST digit classification.

    WEF-Matrix will monitor fc2.weight because fc2 is the
    penultimate hidden layer before the final output layer.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

    def get_wef_weight(self) -> torch.Tensor:
        """Return the layer weight monitored by WEF-Defense."""
        return self.fc2.weight
    