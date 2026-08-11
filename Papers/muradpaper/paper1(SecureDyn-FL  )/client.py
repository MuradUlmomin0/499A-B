import flwr as fl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import SecureDyn1DCNN, compute_dual_loss
from collections import OrderedDict

# 1. Generate Fake N-BaIoT Data for Testing
def load_synthetic_data():
    X = np.random.randn(1000, 115).astype(np.float32)
    y = np.random.randint(0, 11, size=(1000,))
    X_tensor = torch.tensor(X).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

# 2. Define the Flower Client
class SecureDynClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SecureDyn1DCNN()
        self.train_loader = load_synthetic_data()

    def get_parameters(self, config):
        # Send weights to the server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Receive weights from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        # Train locally for 1 epoch
        self.model.train()
        for data, targets in self.train_loader:
            optimizer.zero_grad()
            global_logits, pers_logits = self.model(data)
            loss = compute_dual_loss(global_logits, pers_logits, targets)
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

# 3. Start the Client
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SecureDynClient())