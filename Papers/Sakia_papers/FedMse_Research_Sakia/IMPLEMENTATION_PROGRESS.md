\# FedMSE Implementation Progress



\## Completed Work



\- Integrated the official FedMSE source code.

\- Created the fedmse-implementation Git branch.

\- Created a Python 3.12 virtual environment.

\- Installed and verified the required libraries.

\- Extracted the prepared N-BaIoT dataset.

\- Located and verified the IID 10-client dataset.

\- Verified normal, abnormal, and test\_normal data.

\- Copied the dataset to the path required by FedMSE.

\- Kept the dataset and virtual environment excluded from Git.



\## Environment



\- Python: 3.12.4

\- PyTorch: 2.13.0+cpu

\- Dataset: N-BaIoT IID 10-client

\- Device: CPU
## FedMSE Basic Prototype Implementation

### 1. Environment Setup
- Python 3.12 environment was prepared.
- Required libraries were installed.
- A virtual environment (.venv) was used.

### 2. Dataset Preparation
- N-BaIoT dataset was prepared.
- IID 10-client configuration was used.
- Client-1 to Client-10 were verified.
- Each client contains normal, abnormal, and test_normal traffic data.

### 3. Setup Verification
- Created `verify_fedmse_setup.py`.
- It checks libraries, configuration, dataset, and all 10 clients.
- Setup verification completed successfully.
- Results are stored in:
  - `Outputs/setup_report.txt`
  - `Outputs/setup_summary.json`

### 4. FedMSE Training
The basic FedMSE prototype was trained using:

- Clients: 10
- Client participation: 50%
- Epochs: 5
- Federated rounds: 3
- Aggregation: MSEAvg
- Model: Hybrid SAE-CEN
- Dataset: N-BaIoT IID 10-client

Training code:

`src/main_fedmse_complete.py`

### 5. Federated Learning Process
1. Selected clients trained local models.
2. Local model information was collected.
3. MSEAvg combined the local models.
4. A global model was created.
5. The global model was tested on normal and attack traffic.

### 6. Evaluation Results
Global loss decreased during training:

- Round 1: 3.4503
- Round 2: 2.6878
- Round 3: 2.3635

Client AUC scores were approximately 0.996 to 1.000.

Results are stored in:

- `Outputs/result.json`
- `Outputs/summary.txt`
- `Outputs/fedmse_training_log.txt`

### 7. Current Status
The basic working FedMSE prototype implementation is complete.

Completed:
- Environment setup
- Dataset preparation
- 10-client verification
- Local client training
- MSEAvg aggregation
- Global model creation
- Normal/attack evaluation
- AUC result generation

This is a basic working prototype, not a full reproduction of every large-scale experiment in the FedMSE paper.




