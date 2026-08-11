"""
Basic PEIoT-DS Implementation
Based on: "Federated learning for intrusion detection in IoT environments:
a privacy-preserving strategy" (2025)

What this demo implements:
1. 9 simulated IoT clients
2. Local/private client data
3. A small neural-network intrusion detector
4. Federated Averaging (FedAvg)
5. Federated Averaging with Momentum (FedAvgM)
6. Optional simple Differential Privacy-style noise on client updates
7. Global evaluation after every communication round

This is a BASIC educational reproduction, not the authors' exact full experiment.
It uses synthetic IoT-like data so that it can run immediately.
Replace `make_demo_data()` with N-BaIoT loading for a dataset-based reproduction.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

SEED = 42

# The paper uses nine IoT devices / client datasets.
NUM_CLIENTS = 9

# Keep rounds smaller than the paper's 30 so this basic demo runs quickly.
# Change this to 30 when you want a closer experimental setup.
ROUNDS = 10

# Number of local passes over one client's private data per FL round.
LOCAL_EPOCHS = 1

BATCH_SIZE = 64
LEARNING_RATE = 0.01

# Momentum used by our simple FedAvgM server.
SERVER_MOMENTUM = 0.9

# False = pure FedAvg/FedAvgM demo.
# True = adds small Gaussian noise to the uploaded client parameters.
USE_DP_NOISE = False
DP_NOISE_STD = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# 2. CREATE DEMO IoT NETWORK DATA
# ------------------------------------------------------------

def make_demo_data():
    """
    The real paper uses the N-BaIoT dataset with 115 features.

    To make this file runnable without downloading a huge dataset,
    we create a smaller synthetic binary intrusion dataset.

    Class 0 = benign traffic
    Class 1 = malicious / attack traffic
    """

    X, y = make_classification(
        n_samples=12000,
        n_features=20,
        n_informative=14,
        n_redundant=3,
        n_classes=2,
        weights=[0.45, 0.55],
        class_sep=1.5,
        random_state=SEED,
    )

    # Similar idea to the paper: Min-Max scaling before training.
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)

    # The paper uses a 70% / 30% train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 3. MAKE NON-IID CLIENTS
# ------------------------------------------------------------

def split_into_non_iid_clients(X, y, num_clients=NUM_CLIENTS):
    """
    Federated learning means each client keeps its own local data.

    The paper evaluates non-IID device-specific data.
    Here we create a simple non-IID split:
    each client gets data from a different shuffled portion, and
    some clients are biased toward benign or attack samples.
    """

    rng = np.random.default_rng(SEED)

    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    rng.shuffle(benign_idx)
    rng.shuffle(attack_idx)

    benign_parts = np.array_split(benign_idx, num_clients)
    attack_parts = np.array_split(attack_idx, num_clients)

    clients = []

    for client_id in range(num_clients):
        # Create a mild client-specific imbalance.
        # Even-numbered devices see slightly more attack traffic.
        if client_id % 2 == 0:
            b_take = int(len(benign_parts[client_id]) * 0.70)
            a_take = len(attack_parts[client_id])
        else:
            b_take = len(benign_parts[client_id])
            a_take = int(len(attack_parts[client_id]) * 0.70)

        idx = np.concatenate([
            benign_parts[client_id][:b_take],
            attack_parts[client_id][:a_take],
        ])

        rng.shuffle(idx)

        clients.append((X[idx], y[idx]))

    return clients


# ------------------------------------------------------------
# 4. INTRUSION DETECTION MODEL
# ------------------------------------------------------------

class IntrusionDetector(nn.Module):
    """
    A small neural network for binary intrusion detection.

    The paper's PEIoT-DS architecture discusses deep learning /
    a deep-autoencoder-based intrusion detector. For a basic,
    easy-to-run implementation, this demo uses a compact MLP
    classifier while preserving the federated-learning workflow.
    """

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 5. LOCAL IoT DEVICE TRAINING
# ------------------------------------------------------------

def train_one_client(global_model, X_client, y_client):
    """
    An IoT client receives the current global model,
    trains it only on its PRIVATE local data,
    and returns model parameters.

    Raw client data is never sent to the server.
    """

    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    dataset = TensorDataset(
        torch.tensor(X_client, dtype=torch.float32),
        torch.tensor(y_client, dtype=torch.long),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = local_model(X_batch)

            loss = criterion(predictions, y_batch)

            loss.backward()

            optimizer.step()

    # Only model parameters are returned.
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in local_model.state_dict().items()
    }

    # Educational DP approximation:
    # add small noise to client-uploaded floating-point parameters.
    if USE_DP_NOISE:
        for name in state:
            if torch.is_floating_point(state[name]):
                state[name] += torch.randn_like(state[name]) * DP_NOISE_STD

    return state, len(dataset)


# ------------------------------------------------------------
# 6. FEDERATED AVERAGING
# ------------------------------------------------------------

def fedavg(client_states, client_sizes):
    """
    Weighted Federated Averaging.

    Bigger clients contribute proportionally more to the
    new global model.
    """

    total_samples = sum(client_sizes)

    new_state = copy.deepcopy(client_states[0])

    for name in new_state:
        if torch.is_floating_point(new_state[name]):
            new_state[name].zero_()

            for state, size in zip(client_states, client_sizes):
                weight = size / total_samples
                new_state[name] += state[name] * weight

        else:
            # Non-floating buffers are copied from the first model.
            new_state[name] = client_states[0][name].clone()

    return new_state


# ------------------------------------------------------------
# 7. FEDAVG WITH SERVER MOMENTUM
# ------------------------------------------------------------

def fedavgm(old_global_state, averaged_state, velocity, momentum=SERVER_MOMENTUM):
    """
    Basic FedAvgM idea:

        delta = averaged_client_model - current_global_model
        velocity = momentum * old_velocity + delta
        global = global + velocity

    Momentum remembers part of the previous server update,
    which can make convergence smoother.
    """

    new_global_state = copy.deepcopy(old_global_state)

    for name in old_global_state:

        if not torch.is_floating_point(old_global_state[name]):
            new_global_state[name] = old_global_state[name].clone()
            continue

        delta = averaged_state[name] - old_global_state[name]

        if name not in velocity:
            velocity[name] = torch.zeros_like(delta)

        velocity[name] = momentum * velocity[name] + delta

        new_global_state[name] = old_global_state[name] + velocity[name]

    return new_global_state, velocity


# ------------------------------------------------------------
# 8. TEST GLOBAL MODEL
# ------------------------------------------------------------

def evaluate(model, X_test, y_test):
    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(X_tensor)
        prediction = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, prediction)

    cm = confusion_matrix(y_test, prediction)

    # Matrix layout:
    # [[true benign, false attack],
    #  [missed attack, true attack]]
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn + 1e-12)

    return accuracy, false_positive_rate


# ------------------------------------------------------------
# 9. COMPLETE FEDERATED EXPERIMENT
# ------------------------------------------------------------

def run_federated_learning(method="fedavg"):
    """
    Run one complete FL experiment.

    method:
        "fedavg"  -> Federated Averaging
        "fedavgm" -> Federated Averaging + server momentum
    """

    X_train, X_test, y_train, y_test = make_demo_data()

    clients = split_into_non_iid_clients(
        X_train,
        y_train,
        NUM_CLIENTS,
    )

    input_size = X_train.shape[1]

    global_model = IntrusionDetector(input_size).to(DEVICE)

    velocity = {}

    history = []

    print("\n" + "=" * 65)
    print("PEIoT-DS BASIC FEDERATED LEARNING DEMO")
    print("Aggregation:", method.upper())
    print("IoT clients :", NUM_CLIENTS)
    print("Raw client data sent to server? NO")
    print("=" * 65)

    for round_number in range(1, ROUNDS + 1):

        client_states = []
        client_sizes = []

        # The basic demo uses all nine clients each round.
        # The paper also discusses random client selection.
        selected_clients = list(range(NUM_CLIENTS))

        for client_id in selected_clients:

            X_client, y_client = clients[client_id]

            local_state, local_size = train_one_client(
                global_model,
                X_client,
                y_client,
            )

            client_states.append(local_state)
            client_sizes.append(local_size)

        # First calculate the normal weighted average.
        averaged_state = fedavg(
            client_states,
            client_sizes,
        )

        if method.lower() == "fedavg":

            global_model.load_state_dict(averaged_state)

        elif method.lower() == "fedavgm":

            old_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in global_model.state_dict().items()
            }

            new_state, velocity = fedavgm(
                old_state,
                averaged_state,
                velocity,
            )

            global_model.load_state_dict(new_state)

        else:
            raise ValueError("method must be 'fedavg' or 'fedavgm'")

        global_model = global_model.to(DEVICE)

        accuracy, fpr = evaluate(
            global_model,
            X_test,
            y_test,
        )

        history.append((round_number, accuracy, fpr))

        print(
            f"Round {round_number:02d} | "
            f"Accuracy = {accuracy * 100:6.2f}% | "
            f"FPR = {fpr * 100:6.2f}%"
        )

    return global_model, history


# ------------------------------------------------------------
# 10. MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    # Train once with normal FedAvg.
    fedavg_model, fedavg_history = run_federated_learning("fedavg")

    # Reset seeds so both methods start comparably.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Train again with FedAvgM.
    fedavgm_model, fedavgm_history = run_federated_learning("fedavgm")

    final_avg = fedavg_history[-1]
    final_m = fedavgm_history[-1]

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        f"FedAvg  -> Accuracy: {final_avg[1] * 100:.2f}% | "
        f"FPR: {final_avg[2] * 100:.2f}%"
    )

    print(
        f"FedAvgM -> Accuracy: {final_m[1] * 100:.2f}% | "
        f"FPR: {final_m[2] * 100:.2f}%"
    )

    print("\nImportant:")
    print("- This is a small educational reproduction.")
    print("- The real paper uses N-BaIoT, 9 IoT devices, and 30 FL rounds.")
    print("- For a closer reproduction, replace the synthetic dataset with N-BaIoT.")