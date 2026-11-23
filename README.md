# PRIVATE SHIELD

PRIVATE SHIELD is a CSE499B research project on federated intrusion detection.
It uses an MLP neural network with 46 input features and 34 traffic categories.
The experiments compare FedAvg, adapted FedMADE, SecureDyn-inspired temporal
trust, and an equal blend of temporal and class-utility weights.

The GitHub package includes the code, four saved clean IID seed-42 models,
their validation histories, the final comparison summaries, and a small
20-record prediction demo. These files are enough to run the local dashboard
and the synthetic FedAvg framework check.

## Setup

The pinned environment was tested with Python 3.14.4 on Linux. From the project
folder, create an environment and install the dependencies:

```bash
python3.14 -m venv shield_env
shield_env/bin/python -m pip install -r Requirements.txt
```

If `shield_env` is already installed, use it directly.

## Open the dashboard

```bash
shield_env/bin/python -m streamlit run Support/frontend/app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. The dashboard has four views:

- **Overview:** explains the classifier and the four aggregation methods.
- **Predict traffic:** classifies the bundled records and downloads predictions
  as CSV. The model score is a softmax score, not calibrated confidence.
- **Saved research results:** shows final-test means across three seeds for
  clean and poisoned runs, with IID and non-IID client partitions.
- **Training history:** shows the selected model's saved validation curves.

Predictions run on CPU using already trained weights. Opening the dashboard
does not train a model or score the final test set. The 20 demo records come
from validation data and have no labels in the prediction package; their
predictions are not a new accuracy result.

## Run the small FedAvg check

```bash
shield_env/bin/python main.py --dummy-test \
  --config Support/configs/fedavg_dummy_test.yaml
```

This runs three synthetic clients for three rounds and prints the output
directory. Its metrics verify the framework and are not research results.
For saved-model prediction from the terminal:

```bash
shield_env/bin/python main.py --prepared \
  --config Support/configs/demo_predict.yaml
```

## Research results

The saved final comparison covers four methods, two client partitions, three
seeds, and clean/poisoned conditions: 48 cases in total. The controlled attack
changed DoS-HTTP_Flood labels to BenignTraffic on two of ten simulated clients
in rounds 4–20.

Mean target-recall drop, in percentage points (lower is better):

| Method | IID | Non-IID |
|---|---:|---:|
| FedAvg | 5.72 | 9.49 |
| Adapted FedMADE | 0.10 | 3.75 |
| Temporal trust | 5.58 | 9.25 |
| Equal blend | 1.87 | 6.73 |

Fusion reduced mean damage compared with FedAvg and temporal trust. Adapted
FedMADE had less damage on this measure. Some rare classes had zero recall.
The clients are simulated; device/capture independence cannot be established
from the supplied CSV identities. These methods include documented adaptations
rather than exact reproductions of both papers.

## Files and folders

| Path | Purpose |
|---|---|
| `main.py` | Command-line entry point |
| `Requirements.txt` | Pinned runtime dependencies |
| `Support/frontend/` | Dashboard and prediction service |
| `Support/models/` | MLP architecture and saved-model loading |
| `Support/federated/` | Clients, server and aggregation methods |
| `Support/data/` | Data contracts, loading and preparation code |
| `Support/configs/` | YAML experiment settings |
| `Support/Utils/` | Configuration, logging, seeds and path utilities |
| `Support/tests/` | Software checks |
| `data/processed/demo_predict/` | Small prepared prediction input |
| `Support/results/cp7_v1/analysis/` | Saved final comparison summaries |

Use the capitalized `Support` and `Requirements.txt` paths shown here on Linux.

## Full research artifacts

Large raw datasets, complete prepared training/validation/test partitions,
per-round checkpoints, saved test predictions, and bulk evidence archives are
excluded from Git. Removing them from Git tracking does not delete local files.
The repository therefore supports the lightweight demo without distributing
the complete research bundle.

Full training, full-validation evaluation and offline verification of all 48
test cases need the matching research assets at the paths recorded in their
configs and `Support/runtime_manifest.json`. That manifest retains the full
research inventory and checksums. Full-bundle verification requires those
assets; the dashboard checks only the assets it uses. The `demo_fedavg.yaml`,
`demo_fedmade.yaml`, `demo_temporal.yaml` and `demo_fusion.yaml` configurations
are full-validation evaluation configs, not the small prediction workflow.

## Frontend checks

Install pytest as a development tool, then run:

```bash
shield_env/bin/python -m pip install pytest
shield_env/bin/python -m pytest -q --noconftest \
  Support/tests/test_frontend_service.py Support/tests/test_frontend_app.py \
  -p no:cacheprovider
```

The frontend suite has 33 checks. The existing legacy test fixture imports
lowercase `support`; these self-contained frontend tests use `--noconftest`.
Separate historical preparation tests also require at least 1 GiB of available
RAM. The safeguards remain in place.
