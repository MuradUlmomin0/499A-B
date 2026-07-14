# PRIVATE SHIELD
Privacy-Preserving Federated Intelligence System for IoT Threat Detection

## Week 3 - Person B: Dataset Builder & Clean-up

This folder contains the implementation of the Week 3 Person B deliverable: `dataset_builder.py`.

### 1. What `dataset_builder.py` does
The `dataset_builder.py` script is a command-line tool designed to clean, map, align, and consolidate real N-BaIoT network traffic logs and locally simulated IoT telemetry traffic into a single, unified, machine-learning-ready dataset. 

It handles:
* **File Discovery**: Recursively scans directories to discover all dataset CSV files.
* **Column Normalisation**: Standardises all incoming columns to lowercase `snake_case` and resolves common column name aliases.
* **Label Mapping**: Utilises keyword priorities and file structures to infer class labels (0 = Normal, 1 = DoS, 2 = Port Scan, 3 = Mirai).
* **Data Cleaning**: Drops invalid/missing values, exact duplicates, invalid labels, negative counts, and resolves infinities.
* **Imbalance Analysis**: Computes dataset imbalance ratios and plots distribution checks.
* **Safe SMOTE Balancing**: Splits the data into training/testing sets using stratified splitting *before* applying SMOTE to the training set, completely avoiding data leakage.

---

### 2. Required Input Files
The script expects two folders:
1. **Real Data Directory (`data/N-BaIoT`)**: Containing the real N-BaIoT dataset CSVs in subfolders named after the device (e.g. `danmini_doorbell/benign_traffic.csv`, `danmini_doorbell/mirai_attacks/scan.csv`).
2. **Simulated Data Directory (`data/`)**: Containing locally simulated CSV files (e.g., `normal.csv`, `dos.csv`, `port_scan.csv`, `mirai.csv`).

---

### 3. Required Feature Columns
The final output dataset will contain only the following 10 network feature columns (in this exact order) along with the class label:
1. `flow_duration`
2. `fwd_packets`
3. `bwd_packets`
4. `flow_bytes_per_sec`
5. `flow_pkts_per_sec`
6. `fwd_pkt_len_mean`
7. `bwd_pkt_len_mean`
8. `fin_flag_cnt`
9. `syn_flag_cnt`
10. `rst_flag_cnt`
11. `label` (0 = Normal, 1 = DoS, 2 = Port Scan, 3 = Mirai)

---

### 4. Label Mapping Rules
The N-BaIoT raw files do not contain simple labels. The script automatically infers labels based on file paths, filenames, or existing columns using the following priority order:
* **Mirai (Label 3)**: Matched when path/filename contains the keyword `"mirai"`.
* **Port Scan (Label 2)**: Matched when path/filename contains `"port_scan"` or `"scan"` (unless Mirai priority is matched).
* **DoS (Label 1)**: Matched when path/filename contains `"dos"`, `"flood"`, `"udp"`, `"tcp"`, `"combo"`, or `"junk"`.
* **Normal / Benign (Label 0)**: Matched when path/filename contains `"normal"` or `"benign"`.

---

### 5. Installation Command
Before running the program, install the necessary dependencies:
```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn
```

---

### 6. Run Commands

To run the builder with default settings (no SMOTE, loads files, cleans them, and outputs combined file):
```bash
python dataset_builder.py --real-data-dir data/N-BaIoT --sim-data-dir data --output data/labelled.csv
```

To run the builder with safe SMOTE balancing (generates resampled training set and untouched test set):
```bash
python dataset_builder.py --real-data-dir data/N-BaIoT --sim-data-dir data --output data/labelled.csv --balance smote --test-size 0.20 --random-state 42
```

To run in **strict mode** (stops and exits if any file is missing required columns or cannot be mapped):
```bash
python dataset_builder.py --real-data-dir data/N-BaIoT --sim-data-dir data --strict
```

To limit rows processed per file (e.g. for testing with large N-BaIoT files):
```bash
python dataset_builder.py --real-data-dir data/N-BaIoT --sim-data-dir data --max-rows-per-file 1000
```

---

### 7. Output Files
The builder will produce:
* `data/labelled.csv`: The main combined dataset containing only the 10 feature columns and the `label` column (naturally distributed).
* `data/labelled_audit.csv`: A duplicate of the combined dataset including metadata columns (like `source`, `filename`, `timestamp`, `topic`, `device_id`, and `attack_name`) for verification and logging.
* `data/dataset_summary.json`: A summary JSON file containing counts of loaded files, rows removed, class distributions, imbalance ratios, and output paths.

If `--balance smote` is run:
* `data/train_original.csv`: The training features split from the dataset (unbalanced).
* `data/train_smote.csv`: The training features after applying SMOTE balancing (all classes will have equal samples).
* `data/test.csv`: The testing features (naturally distributed, untouched) to evaluate ML models correctly.

---

### 8. How SMOTE is Safely Applied
To prevent **data leakage** (where synthetic information from the test set leaks into training, causing overly optimistic accuracy scores):
1. The combined dataset is split into training and testing partitions using a stratified split (retaining class percentages).
2. The testing partition (`data/test.csv`) is saved immediately and **never** exposed to SMOTE.
3. SMOTE is applied *only* to the training partition (`X_train`, `y_train`) to generate the synthetic balanced dataset (`data/train_smote.csv`).

---

### 9. Dependency on Person A's Week 3 Work
If the simulated files (like `normal.csv`) only contain IoT sensor telemetry columns (such as `temperature`, `humidity`, `packets_per_sec`, `bytes_per_pkt`, `port`), the script will log warnings indicating that **Person A's Week 3 feature-generation update is still required**. The logger/simulator must be updated to output the 10 network flow columns (like `flow_duration`, `fwd_packets`, `bwd_packets`, etc.) for simulated data to align properly with N-BaIoT features.

---

### 10. Common Errors and Solutions

* **ImportError: No module named 'imblearn'**:
  * *Reason*: The `imbalanced-learn` library is not installed.
  * *Solution*: Run `pip install imbalanced-learn`.
* **ValueError: Expected n_neighbors <= n_samples**:
  * *Reason*: A class has too few samples (less than 6) to run SMOTE.
  * *Solution*: The script automatically adapts `k_neighbors = min(5, smallest_class_count - 1)`. If the smallest class has only 1 sample, it will output a message and skip SMOTE gracefully.
* **Strict mode error on missing features**:
  * *Reason*: A CSV does not contain all required columns, and `--strict` is enabled.
  * *Solution*: Ensure Person A's script has generated the correct features, or run without `--strict` to automatically skip incompatible files.