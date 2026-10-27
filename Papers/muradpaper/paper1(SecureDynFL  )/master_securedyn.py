import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from phe import paillier

# ==========================================
# 1. ARCHITECTURE & DATA
# ==========================================
class SecureDyn1DCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SecureDyn1DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(256, 128, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2)
        )
        self.global_classifier = nn.Linear(32, num_classes)
        self.personalized_classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.global_classifier(features), self.personalized_classifier(features), features

def generate_non_iid_data(num_clients=20, num_samples=5000, num_classes=11, eta=0.1):
    X = np.random.randn(num_samples, 115).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    client_datasets = []
    for _ in range(num_clients):
        indices = np.random.choice(num_samples, size=int(num_samples/num_clients), replace=False)
        dataset = TensorDataset(torch.tensor(X[indices]).unsqueeze(1), torch.tensor(y[indices], dtype=torch.long))
        client_datasets.append(DataLoader(dataset, batch_size=32, shuffle=True))
    return client_datasets

# ==========================================
# 2. CLIENT LOGIC (Training, Pruning, Encrypting)
# ==========================================
class IoTClient:
    def __init__(self, client_id, dataset, is_malicious=False):
        self.id = client_id
        self.dataset = dataset
        self.is_malicious = is_malicious
        self.local_model = SecureDyn1DCNN()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.01)

    def train_epoch(self):
        self.local_model.train()
        for data, targets in self.dataset:
            if self.is_malicious:
                targets = torch.zeros_like(targets) # Poisoning: Label Flipping
            
            self.optimizer.zero_grad()
            g_logits, p_logits, _ = self.local_model(data)
            
            # Dual Loss
            loss_ce = F.cross_entropy(g_logits, targets)
            class_counts = torch.bincount(targets, minlength=11).float()
            alpha_y = (class_counts / targets.size(0)) + 1e-7
            loss_la = F.cross_entropy(p_logits + torch.log(alpha_y).unsqueeze(0), targets)
            
            (loss_ce + loss_la).backward()
            self.optimizer.step()

    def apply_efficiency(self):
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if "feature_extractor" in name or "global" in name:
                    # Clip & Prune
                    mean_val = torch.mean(torch.abs(param))
                    param.clamp_(-2.0 * mean_val, 2.0 * mean_val)
                    threshold = np.percentile(torch.abs(param).cpu().numpy(), 10)
                    param[torch.abs(param) < threshold] = 0.0

    def get_upload_package(self, public_key):
        """ Uploads plaintext feature extractor, but ENCRYPTS the global classifier """
        upload_dict = {}
        state = self.local_model.state_dict()
        for name, param in state.items():
            if "feature_extractor" in name:
                upload_dict[name] = param.clone()
            elif "global_classifier" in name:
                # Encrypt the classifier
                flat_weights = param.view(-1).cpu().tolist()
                upload_dict[name] = [public_key.encrypt(x) for x in flat_weights]
        return upload_dict

# ==========================================
# 3. SERVER LOGIC (Auditor & Secure Aggregation)
# ==========================================
def central_auditor(upload_packages):
    """ Detects poisoned clients by looking at their plaintext feature extractors """
    flat_updates = []
    for pkg in upload_packages:
        # Extract just the plaintext parts to audit their behavior
        flat_tensor = torch.cat([param.view(-1) for name, param in pkg.items() if "feature_extractor" in name])
        flat_updates.append(flat_tensor.cpu().numpy())
    
    flat_updates = np.array(flat_updates)
    
    # PCA + GMM
    pca = PCA(n_components=5, random_state=42)
    compressed = pca.fit_transform(flat_updates)
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', reg_covar=1e-3, random_state=42)
    labels = gmm.fit_predict(compressed)
    
    benign_id = 0 if np.sum(labels == 0) > np.sum(labels == 1) else 1
    
    accepted = []
    rejected = []
    for i, label in enumerate(labels):
        if label == benign_id:
            accepted.append(upload_packages[i])
        else:
            rejected.append(i)
            
    if rejected:
        print(f"   🚨 AUDITOR: Blocked poisoned clients {rejected}")
    return accepted

def secure_aggregation(accepted_pkgs, global_model, private_key):
    """ Averages plaintext feature extractors and homomorphically averages encrypted classifiers """
    num_clients = len(accepted_pkgs)
    global_dict = global_model.state_dict()
    
    # 1. Plaintext Aggregation (Feature Extractor)
    for name in global_dict.keys():
        if "feature_extractor" in name:
            global_dict[name] = sum([pkg[name] for pkg in accepted_pkgs]) / num_clients
            
    # 2. Encrypted Aggregation (Global Classifier)
    for name in global_dict.keys():
        if "global_classifier" in name:
            shape = global_dict[name].shape
            summed_enc = list(accepted_pkgs[0][name])
            
            # Homomorphic Addition
            for pkg in accepted_pkgs[1:]:
                for i in range(len(summed_enc)):
                    summed_enc[i] = summed_enc[i] + pkg[name][i]
            
            # Decrypt and apply average
            avg_decrypted = [private_key.decrypt(x) * (1.0 / num_clients) for x in summed_enc]
            global_dict[name] = torch.tensor(avg_decrypted).view(shape)
            
    global_model.load_state_dict(global_dict)

# ==========================================
# 4. MASTER EXECUTION
# ==========================================
def run_secure_dyn_fl():
    print("=== SECUREDYN-FL MASTER PIPELINE ===")
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=512)
    datasets = generate_non_iid_data(num_clients=10)
    
    # Create 10 clients (Clients 8 & 9 are Attackers)
    clients = [IoTClient(i, datasets[i], is_malicious=(i >= 8)) for i in range(10)]
    server_model = SecureDyn1DCNN()
    
    for round_num in range(1, 4):
        print(f"\n--- Round {round_num}/3 ---")
        start = time.time()
        
        # 1. Local Training & Encryption
        print("   [Clients] Training, Pruning, and Encrypting...")
        uploads = []
        for c in clients:
            c.local_model.load_state_dict(server_model.state_dict(), strict=False)
            c.train_epoch()
            c.apply_efficiency()
            uploads.append(c.get_upload_package(pub_key))
            
        # 2. Central Auditor
        verified_uploads = central_auditor(uploads)
        
        # 3. Secure Server Aggregation
        print("   [Server] Executing Homomorphic Aggregation...")
        secure_aggregation(verified_uploads, server_model, priv_key)
        
        print(f"✅ Round {round_num} Complete in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    run_secure_dyn_fl()