import csv
import os
import random
from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from client import (
    create_free_rider_update,
    train_honest_client,
)
from data_loader import create_client_loaders
from model import LeNet
from server import detect_free_riders, federated_average


ModelState = Dict[str, torch.Tensor]


NUM_CLIENTS = 5
FREE_RIDER_IDS = {4}

GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1

BATCH_SIZE = 64
LEARNING_RATE = 0.01

DEVICE = "cpu"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def copy_state(
    state: ModelState,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in state.items()
    )


def evaluate_model(
    model_state: ModelState,
    test_loader: DataLoader,
    device: str,
) -> float:
    """Calculate model accuracy on test data."""

    model = LeNet().to(device)
    model.load_state_dict(model_state)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted_labels = predictions.argmax(dim=1)

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def save_history(history: list[dict]) -> None:
    os.makedirs("results", exist_ok=True)

    csv_path = "results/history.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "round",
                "accuracy",
                "actual_free_rider",
                "detected_free_rider",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    rounds = [row["round"] for row in history]
    accuracies = [
        row["accuracy"] for row in history
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(rounds, accuracies, marker="o")
    plt.xlabel("Global Round")
    plt.ylabel("Honest Global Model Accuracy (%)")
    plt.title("WEF-Defense Prototype Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "results/accuracy_by_round.png",
        dpi=200,
    )
    plt.close()


def main() -> None:
    set_seed(SEED)

    print("Loading MNIST dataset...")

    client_loaders, test_loader = (
        create_client_loaders(
            num_clients=NUM_CLIENTS,
            batch_size=BATCH_SIZE,
        )
    )

    global_model = LeNet()
    global_state = copy_state(
        global_model.state_dict()
    )

    history: list[dict] = []

    for round_number in range(
        1,
        GLOBAL_ROUNDS + 1,
    ):
        print("\n" + "=" * 55)
        print(f"Global Round: {round_number}")
        print("=" * 55)

        client_updates = []
        client_wefs = []

        for client_id in range(NUM_CLIENTS):
            if client_id in FREE_RIDER_IDS:
                print(
                    f"Client {client_id}: "
                    "Free-Rider — no real training"
                )

                update, wef, loss = (
                    create_free_rider_update(
                        global_state
                    )
                )

            else:
                print(
                    f"Client {client_id}: "
                    "honest local training"
                )

                update, wef, loss = (
                    train_honest_client(
                        global_state=global_state,
                        data_loader=(
                            client_loaders[client_id]
                        ),
                        local_epochs=LOCAL_EPOCHS,
                        learning_rate=(
                            LEARNING_RATE
                        ),
                        device=DEVICE,
                    )
                )

                print(
                    f"Client {client_id} "
                    f"average loss: {loss:.4f}"
                )

            client_updates.append(update)
            client_wefs.append(wef)

        detected_free_riders, scores = (
            detect_free_riders(
                client_wefs,
                expected_free_riders=len(
                    FREE_RIDER_IDS
                ),
            )
        )

        detected_set = set(
            detected_free_riders
        )

        honest_client_ids = [
            client_id
            for client_id in range(NUM_CLIENTS)
            if client_id not in detected_set
        ]

        print("\nClient suspicion scores:")

        for client_id, score in enumerate(scores):
            print(
                f"Client {client_id}: "
                f"{score.item():.4f}"
            )

        print(
            "\nActual Free-Rider IDs:",
            sorted(FREE_RIDER_IDS),
        )

        print(
            "Detected Free-Rider IDs:",
            sorted(detected_free_riders),
        )

        print(
            "Clients used for honest aggregation:",
            honest_client_ids,
        )

        honest_updates = [
            client_updates[client_id]
            for client_id in honest_client_ids
        ]

        global_state = federated_average(
            honest_updates
        )

        accuracy = evaluate_model(
            global_state,
            test_loader,
            DEVICE,
        )

        print(
            f"Honest global model accuracy: "
            f"{accuracy:.2f}%"
        )

        history.append(
            {
                "round": round_number,
                "accuracy": round(
                    accuracy,
                    2,
                ),
                "actual_free_rider": str(
                    sorted(FREE_RIDER_IDS)
                ),
                "detected_free_rider": str(
                    sorted(
                        detected_free_riders
                    )
                ),
            }
        )

    save_history(history)

    print("\nImplementation completed.")
    print(
        "Results saved inside the results folder."
    )


if __name__ == "__main__":
    main()
    