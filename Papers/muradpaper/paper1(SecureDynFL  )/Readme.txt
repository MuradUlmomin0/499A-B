Note: This implementation uses the phe library (Python Partially Homomorphic Encryption) to seamlessly handle Paillier encryption across modern Python versions.)

⚙️ Usage
To execute the full SecureDyn-FL pipeline, simply run the master script:

Bash
python master_securedyn.py
Expected Output:
When executed, the script will:

Initialize a simulated highly heterogeneous (Non-IID) IoT network.

Train the decoupled neural networks locally on each device.

Apply L1-norm pruning and Paillier encryption.

Trigger a server-side Auditor Alert, successfully catching and rejecting injected malicious attackers (e.g., label-flipping nodes).

Output the execution time of the secure homomorphic aggregation round.

📚 Reference
If you utilize this code or extend upon it, please refer to the foundational concepts presented in the original paper:

Soomro, I. A., et al. (2026). SecureDyn-FL: A Robust Privacy-Preserving Federated Learning Framework for Intrusion Detection in IoT Networks. arXiv:2601.06466v1.
