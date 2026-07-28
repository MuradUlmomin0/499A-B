import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Import the architecture and data generator from Phase 1
from model import SecureDyn1DCNN
from data_setup import generate_non_iid_data

class IoTClient:
    def __init__(self, client_id, dataset):
        self.id = client_id
        self.dataset = dataset
        
        # Each client gets its own persistent local model. 
        # This ensures the personalized head is saved between rounds.
        self.local_model = SecureDyn1DCNN(num_classes=11)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.9)
    
    def sync_with_server(self, global_state_dict):
        """
        Downloads the global feature extractor and global head from the server.
        Leaves the personalized head untouched.
        """
        local_state = self.local_model.state_dict()
        for key in global_state_dict.keys():
            # Strict conditional check to enforce the model decoupling strategy
            if "feature_extractor" in key or "global_classifier" in key:
                local_state[key] = global_state_dict[key]
        self.local_model.load_state_dict(local_state)

    def train_local_epoch(self):
        """Executes the dual-objective loss training locally."""
        self.local_model.train()
        for data, targets in self.dataset:
            self.optimizer.zero_grad()
            global_logits, pers_logits, _ = self.local_model(data)
            
            # Global Cross-Entropy
            loss_ce = F.cross_entropy(global_logits, targets)
            
            # Logit-Adjusted Personalization Loss
            class_counts = torch.bincount(targets, minlength=11).float()
            alpha_y = (class_counts / targets.size(0)) + 1e-7
            adjusted_logits = pers_logits + torch.log(alpha_y).unsqueeze(0)
            loss_la = F.cross_entropy(adjusted_logits, targets)
            
            total_loss = loss_ce + loss_la
            total_loss.backward()
            self.optimizer.step()
            
    def apply_clipping_and_pruning(self, alpha=2.0, prune_percentile=10):
        """
        Applies layer-wise clipping and unstructured L1-norm pruning.
        This stabilizes the training and introduces sparsity for efficiency.
        """
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if "feature_extractor" in name or "global_classifier" in name:
                    
                    # 1. Dynamic Mean Clipping
                    mean_val = torch.mean(torch.abs(param))
                    param.clamp_(-alpha * mean_val, alpha * mean_val)
                    
                    # 2. Unstructured L1 Pruning
                    threshold = np.percentile(torch.abs(param).cpu().numpy(), prune_percentile)
                    param[torch.abs(param) < threshold] = 0.0
                    
    def get_upload_weights(self):
        """Extracts only the globally shared parameters for server upload."""
        upload_dict = {}
        for name, param in self.local_model.state_dict().items():
            if "feature_extractor" in name or "global_classifier" in name:
                upload_dict[name] = param.clone()
        return upload_dict


def federated_averaging(global_model, client_weights_list):
    """
    Standard FedAvg algorithm: Aggregates client updates by computing the mathematical average.
    """
    global_dict = global_model.state_dict()
    
    # Initialize zero tensors for summing the weights
    averaged_dict = {key: torch.zeros_like(val) for key, val in global_dict.items() 
                     if "feature_extractor" in key or "global_classifier" in key}
    
    num_clients = len(client_weights_list)
    
    # Sum the weights from all clients
    for client_weights in client_weights_list:
        for key in averaged_dict.keys():
            averaged_dict[key] += client_weights[key] / float(num_clients)
            
    # Apply the aggregated weights back to the central global model
    for key in averaged_dict.keys():
        global_dict[key] = averaged_dict[key]
        
    global_model.load_state_dict(global_dict)


def run_phase_2():
    num_rounds = 5
    num_clients = 20
    
    print("\n--- Initializing Phase 2 Federated Infrastructure ---")
    
    # 1. Generate Non-IID data and initialize 20 distinct clients
    datasets = generate_non_iid_data(num_clients=num_clients)
    clients = [IoTClient(i, datasets[i]) for i in range(num_clients)]
    
    # 2. Initialize the Central Server Global Model
    server_model = SecureDyn1DCNN(num_classes=11)
    
    print("\n--- Starting Federated Training Loop ---")
    for fl_round in range(num_rounds):
        print(f"Executing Communication Round {fl_round + 1}/{num_rounds}...")
        
        global_state = server_model.state_dict()
        uploaded_weights = []
        
        for client in clients:
            # Step A: Download the global model
            client.sync_with_server(global_state)
            
            # Step B: Train on local non-IID data
            client.train_local_epoch()
            
            # Step C: Apply defensive pruning and clipping
            client.apply_clipping_and_pruning()
            
            # Step D: Upload shared layers securely
            uploaded_weights.append(client.get_upload_weights())
            
        # Step E: Server aggregates all uploaded client weights
        federated_averaging(server_model, uploaded_weights)
        print(f"✅ Round {fl_round + 1} Aggregation Complete.")
        
    print("\n Complete! The mock federated environment is strictly functional.")

if __name__ == "__main__":
    run_phase_2()