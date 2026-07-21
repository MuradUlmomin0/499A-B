import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from sklearn.decomposition import PCA

# Import from Phase 1
from model import SecureDyn1DCNN
from data_setup import generate_non_iid_data
from federated_loop import federated_averaging # Reusing our averaging function

class IoTClient:
    def __init__(self, client_id, dataset, is_malicious=False):
        self.id = client_id
        self.dataset = dataset
        self.is_malicious = is_malicious # NEW: Attacker flag
        self.local_model = SecureDyn1DCNN(num_classes=11)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.9)
    
    def sync_with_server(self, global_state_dict):
        local_state = self.local_model.state_dict()
        for key in global_state_dict.keys():
            if "feature_extractor" in key or "global_classifier" in key:
                local_state[key] = global_state_dict[key]
        self.local_model.load_state_dict(local_state)

    def train_local_epoch(self):
        self.local_model.train()
        for data, targets in self.dataset:
            
            # --- THE POISONING ATTACK ---
            if self.is_malicious:
                # Label-Flipping: The attacker forces all target labels to 0 (Benign).
                # This teaches the model to ignore real attacks.
                targets = torch.zeros_like(targets)
                
            self.optimizer.zero_grad()
            global_logits, pers_logits, _ = self.local_model(data)
            
            loss_ce = F.cross_entropy(global_logits, targets)
            class_counts = torch.bincount(targets, minlength=11).float()
            alpha_y = (class_counts / targets.size(0)) + 1e-7
            adjusted_logits = pers_logits + torch.log(alpha_y).unsqueeze(0)
            loss_la = F.cross_entropy(adjusted_logits, targets)
            
            total_loss = loss_ce + loss_la
            total_loss.backward()
            self.optimizer.step()
            
    def get_upload_weights(self):
        upload_dict = {}
        for name, param in self.local_model.state_dict().items():
            if "feature_extractor" in name or "global_classifier" in name:
                upload_dict[name] = param.clone()
        return upload_dict


# ==========================================
# NEW: THE CENTRAL AUDITOR (GMM + MD)
# ==========================================
def central_auditor_filter(uploaded_weights_list):
    """
    Analyzes client gradient updates using PCA, GMM clustering, and Mahalanobis distance.
    Drops malicious clients before they can poison the global model.
    """
    print("   -> Auditor analyzing incoming weights...")
    
    # 1. Flatten the weights for each client into a 1D numerical array
    flattened_updates = []
    for client_weights in uploaded_weights_list:
        flat_tensor = torch.cat([param.view(-1) for param in client_weights.values()])
        flattened_updates.append(flat_tensor.cpu().numpy())
        
    flattened_updates = np.array(flattened_updates)
    
    # ==========================================
    # NEW: APPLY PCA COMPRESSION
    # Compress 126,539 parameters down to 10 core principal components
    # to prevent RAM overload and mathematical singularity.
    # ==========================================
    pca = PCA(n_components=10, random_state=42)
    compressed_updates = pca.fit_transform(flattened_updates)
    
    # 2. Fit the Gaussian Mixture Model on the COMPRESSED data
    gmm = GaussianMixture(n_components=2, covariance_type='full', reg_covar=1e-3, random_state=42)
    labels = gmm.fit_predict(compressed_updates)
    
    # 3. Identify the "Benign" cluster (Assume the majority of clients are honest)
    cluster_0_count = np.sum(labels == 0)
    cluster_1_count = np.sum(labels == 1)
    benign_cluster_id = 0 if cluster_0_count > cluster_1_count else 1
    
    benign_mean = gmm.means_[benign_cluster_id]
    benign_cov = gmm.covariances_[benign_cluster_id]
    
    # Compute inverse covariance for Mahalanobis Distance
    try:
        inv_cov = np.linalg.inv(benign_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(benign_cov)
    
    # 4. Filter clients based on Mahalanobis Distance threshold
    accepted_weights = []
    rejected_client_indices = []
    
    # Note: We iterate over the COMPRESSED updates for distance checking
    for i, client_update in enumerate(compressed_updates):
        md = distance.mahalanobis(client_update, benign_mean, inv_cov)
        
        # Threshold heuristic: If they were clustered wrong, reject.
        if labels[i] == benign_cluster_id:
            accepted_weights.append(uploaded_weights_list[i])
        else:
            rejected_client_indices.append(i)
            
    if rejected_client_indices:
        print(f"   🚨 AUDITOR ALERT: Dropped poisoned updates from clients: {rejected_client_indices}")
    
    return accepted_weights


def run_phase_3():
    num_rounds = 5
    num_clients = 20
    
    print("\n--- Initializing Phase 3: SecureDyn-FL with Active Auditor ---")
    datasets = generate_non_iid_data(num_clients=num_clients)
    
    # Initialize Clients (Force clients 16, 17, 18, and 19 to be Malicious)
    clients = []
    for i in range(num_clients):
        is_malicious = True if i >= 16 else False
        clients.append(IoTClient(i, datasets[i], is_malicious=is_malicious))
        
    server_model = SecureDyn1DCNN(num_classes=11)
    
    for fl_round in range(num_rounds):
        print(f"\nExecuting Communication Round {fl_round + 1}/{num_rounds}...")
        
        global_state = server_model.state_dict()
        uploaded_weights = []
        
        for client in clients:
            client.sync_with_server(global_state)
            client.train_local_epoch()
            
            # Note: Skipping pruning here just to keep the script concise and focused on auditing
            uploaded_weights.append(client.get_upload_weights())
            
        # --- THE AUDIT ---
        # Instead of trusting everyone, we pass weights through the filter first
        verified_weights = central_auditor_filter(uploaded_weights)
        
        # --- AGGREGATION ---
        # We only aggregate the safe weights
        federated_averaging(server_model, verified_weights)
        print(f"✅ Round {fl_round + 1} Complete. Aggregated {len(verified_weights)}/{num_clients} clients.")

if __name__ == "__main__":
    run_phase_3()