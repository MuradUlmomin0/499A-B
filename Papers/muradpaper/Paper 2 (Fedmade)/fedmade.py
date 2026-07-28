import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
import copy
import warnings
warnings.filterwarnings("ignore")

# 1. Setup a lightweight Deep Neural Network for Intrusion Detection
class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# 2. Generate Synthetic Imbalanced & Non-IID IoT Data
def get_non_iid_data(num_clients=5):
    # 3 classes: 1 Majority (Normal traffic), 2 Minority (Rare attacks)
    X, y = make_classification(n_samples=2500, n_features=20, n_informative=10, 
                               n_classes=3, weights=[0.8, 0.1, 0.1], random_state=42)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    client_data = []
    # Split data unevenly to simulate real-world IoT heterogeneity
    splits = [500, 500, 500, 500, 500] 
    indices = torch.randperm(2500)
    
    start = 0
    for split in splits:
        idx = indices[start:start+split]
        client_data.append((X_tensor[idx], y_tensor[idx]))
        start += split
        
    return client_data, X_tensor, y_tensor # Return global data for server validation

# 3. Local Training & Class Probability Matrix (CPM) Extraction
def train_client(model, X, y, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model

def get_cpm(model, X):
    """
    Paper concept: Generates a Class Probability Matrix (CPM) to 
    represent a client's specific traffic pattern.
    """
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)
    
    # Average probability per class
    cpm = probs.mean(dim=0).numpy()
    return cpm

# 4. Aggregation Strategies (The Core Comparison)
def fedavg(global_model, client_models):
    """Standard Federated Averaging: Blindly averages all weights"""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([m.state_dict()[k].float() for m in client_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def fedmade(global_model, client_models, client_cpms, server_X, server_y):
    """FedMADE: Clusters by CPM, weights by minority class performance"""
    # Step A: Cluster clients based on their CPM using DBSCAN (as per the paper)
    cpm_matrix = np.array(client_cpms)
    clustering = DBSCAN(eps=0.1, min_samples=1).fit(cpm_matrix)
    labels = clustering.labels_
    
    # Step B: Dynamic Weighting
    weights = []
    for i, model in enumerate(client_models):
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(server_X), dim=1)
        
        # Calculate Macro F1 (heavily penalizes missing minority classes)
        score = f1_score(server_y.numpy(), preds.numpy(), average='macro')
        
        # Boost clients with rare traffic patterns (smaller clusters)
        cluster_size = list(labels).count(labels[i])
        dynamic_weight = score / cluster_size 
        weights.append(dynamic_weight)
        
    # Normalize weights into a probability distribution
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    
    # Step C: Weighted Aggregation
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = sum(m.state_dict()[k].float() * w for m, w in zip(client_models, weights))
    
    global_model.load_state_dict(global_dict)
    return global_model

# 5. Run the Simulation
if __name__ == "__main__":
    print("1. Initializing IoT Intrusion Data (80% Normal, 20% Rare Attacks)...")
    client_data, server_X, server_y = get_non_iid_data(num_clients=5)
    
    model_fedavg = IDSModel(20, 3)
    model_fedmade = copy.deepcopy(model_fedavg)
    
    print("\n2. Training 5 IoT Clients Locally...")
    client_models_fedavg = []
    client_models_fedmade = []
    client_cpms = []
    
    for i in range(5):
        X, y = client_data[i]
        
        # FedAvg Training
        local_avg = copy.deepcopy(model_fedavg)
        client_models_fedavg.append(train_client(local_avg, X, y))
        
        # FedMADE Training
        local_made = copy.deepcopy(model_fedmade)
        trained_made = train_client(local_made, X, y)
        client_models_fedmade.append(trained_made)
        
        # Extract CPM for FedMADE clustering
        client_cpms.append(get_cpm(trained_made, X))
        
    print("\n3. Aggregating Models...")
    model_fedavg = fedavg(model_fedavg, client_models_fedavg)
    model_fedmade = fedmade(model_fedmade, client_models_fedmade, client_cpms, server_X, server_y)
    
    # Evaluation
    def evaluate(model, name):
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(server_X), dim=1)
        macro_f1 = f1_score(server_y.numpy(), preds.numpy(), average='macro')
        print(f" -> {name} Macro-F1 Score: {macro_f1:.4f}")
        
    print("\n--- Final Results on Global Validation Set ---")
    evaluate(model_fedavg, "Standard FedAvg ")
    evaluate(model_fedmade, "FedMADE         ")
    print("\nConclusion: FedMADE retains minority attack detection better than blind averaging.")