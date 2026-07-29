from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import LeNet
from wef import initialize_wef, update_wef_matrix


ModelState = Dict[str, torch.Tensor]


def copy_model_state(
    model: nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    """Copy model parameters safely to CPU."""

    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in model.state_dict().items()
    )


def train_honest_client(
    global_state: ModelState,
    data_loader: DataLoader,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    device: str = "cpu",
) -> Tuple[ModelState, torch.Tensor, float]:
    """
    Train an honest client using real local data.

    Returns:
    - updated model weights
    - WEF-Matrix
    - average training loss
    """

    model = LeNet().to(device)
    model.load_state_dict(global_state)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )

    criterion = nn.CrossEntropyLoss()

    wef_matrix = initialize_wef(
        model.get_wef_weight()
    )

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            previous_weight = (
                model.get_wef_weight()
                .detach()
                .clone()
            )

            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            current_weight = (
                model.get_wef_weight()
                .detach()
                .clone()
            )

            wef_matrix = update_wef_matrix(
                wef_matrix,
                previous_weight,
                current_weight,
            )

            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / max(total_batches, 1)

    return (
        copy_model_state(model),
        wef_matrix,
        average_loss,
    )


def create_free_rider_update(
    global_state: ModelState,
) -> Tuple[ModelState, torch.Tensor, float]:
    """
    Ordinary Free-Rider attack.

    The client performs no real training.
    It sends the unchanged global model and a zero WEF-Matrix.
    """

    fake_state = OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in global_state.items()
    )

    fake_wef = torch.zeros_like(
        global_state["fc2.weight"],
        dtype=torch.float32,
    )

    fake_loss = 0.0

    return fake_state, fake_wef, fake_loss