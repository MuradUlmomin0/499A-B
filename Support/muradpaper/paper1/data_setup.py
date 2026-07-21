import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_non_iid_data(num_clients=20, num_samples=10000, num_features=115, num_classes=11, eta=0.1):
    """
    Generates synthetic network traffic and divides it unevenly among clients
    using a Dirichlet distribution to simulate real-world IoT heterogeneity.
    """
    print("Generating simulated N-BaIoT network traffic...")
    
    # Create random features and random attack labels
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Use Dirichlet distribution to create skewed data proportions for each client
    class_priors = np.random.dirichlet(alpha=[eta] * num_classes, size=num_clients)
    
    client_datasets = []
    
    # Distribute the data to clients based on the skewed proportions
    for client_id in range(num_clients):
        # Grab a random subset of data for this specific client
        indices = np.random.choice(num_samples, size=int(num_samples/num_clients), replace=False)
        X_client = torch.tensor(X[indices]).unsqueeze(1) # Add channel dimension for CNN
        y_client = torch.tensor(y[indices], dtype=torch.long)
        
        # Package into a PyTorch DataLoader
        dataset = TensorDataset(X_client, y_client)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        client_datasets.append(loader)
        
    print(f"Successfully created {num_clients} non-IID client datasets.")
    return client_datasets

if __name__ == "__main__":
    # Test the function
    datasets = generate_non_iid_data()