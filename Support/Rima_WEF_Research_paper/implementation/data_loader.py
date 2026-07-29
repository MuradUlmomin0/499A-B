from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_client_loaders(
    num_clients: int = 5,
    batch_size: int = 64,
    train_limit: int = 10000,
    test_limit: int = 2000,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Download MNIST and divide training data among clients.

    This first implementation uses IID distribution:
    every client receives randomly selected MNIST samples.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(seed)

    train_indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:train_limit]

    client_splits = torch.chunk(train_indices, num_clients)

    client_loaders: List[DataLoader] = []

    for split in client_splits:
        subset = Subset(train_dataset, split.tolist())

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
        )

        client_loaders.append(loader)

    test_indices = list(range(min(test_limit, len(test_dataset))))
    test_subset = Subset(test_dataset, test_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return client_loaders, test_loader
