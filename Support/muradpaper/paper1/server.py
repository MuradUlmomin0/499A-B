import flwr as fl

# Define the Aggregation Strategy
# In the future, you will inject your Scikit-Learn GMM logic here to filter bad clients
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Train on all connected clients
    min_fit_clients=2, # Wait for 2 clients to connect before starting
    min_available_clients=2,
)

if __name__ == "__main__":
    print("Starting SecureDyn-FL Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # Run for 3 federated rounds
        strategy=strategy,
    )