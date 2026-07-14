#!/usr/bin/env python3
"""
PRIVATE SHIELD - Week 3 Verification and Test Suite
Generates temporary mock data with appropriate feature columns and column aliases,
then executes dataset_builder.py to verify its functionality.
"""

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MOCK_DIR = BASE_DIR / "data" / "mock_temp"
REAL_MOCK_DIR = MOCK_DIR / "N-BaIoT"
SIM_MOCK_DIR = MOCK_DIR / "simulated"

OUTPUT_FILE = MOCK_DIR / "labelled.csv"
AUDIT_FILE = MOCK_DIR / "labelled_audit.csv"
SUMMARY_FILE = MOCK_DIR / "dataset_summary.json"

REQUIRED_FEATURES = [
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt"
]

def generate_mock_row(label: int, has_aliases: bool = False, missing_cols: list = None) -> dict:
    """Generate a single mock row of network telemetry with specified label."""
    if missing_cols is None:
        missing_cols = []
        
    # Set feature distributions based on label to simulate real patterns
    if label == 0:  # Normal
        dur = np.random.uniform(1.0, 10.0)
        fwd = np.random.randint(1, 20)
        bwd = np.random.randint(1, 10)
        pkts_s = fwd + bwd / dur
        bytes_s = pkts_s * np.random.randint(32, 256)
        fwd_mean = np.random.randint(32, 256)
        bwd_mean = np.random.randint(32, 256)
        fin = np.random.randint(0, 2)
        syn = np.random.randint(0, 2)
        rst = np.random.randint(0, 2)
    elif label == 1:  # DoS
        dur = np.random.uniform(0.1, 1.5)
        fwd = np.random.randint(500, 2000)
        bwd = np.random.randint(0, 5)
        pkts_s = fwd + bwd / dur
        bytes_s = pkts_s * np.random.randint(800, 1500)
        fwd_mean = np.random.randint(800, 1500)
        bwd_mean = np.random.randint(0, 50)
        fin = 0
        syn = np.random.randint(100, 500)
        rst = np.random.randint(20, 100)
    elif label == 2:  # Port Scan
        dur = np.random.uniform(0.1, 1.0)
        fwd = np.random.randint(50, 300)
        bwd = np.random.randint(0, 10)
        pkts_s = fwd + bwd / dur
        bytes_s = pkts_s * np.random.randint(40, 100)
        fwd_mean = np.random.randint(40, 100)
        bwd_mean = np.random.randint(0, 40)
        fin = np.random.randint(5, 50)
        syn = np.random.randint(50, 300)
        rst = np.random.randint(50, 300)
    else:  # Mirai
        dur = np.random.uniform(0.5, 5.0)
        fwd = np.random.randint(200, 1000)
        bwd = np.random.randint(0, 20)
        pkts_s = fwd + bwd / dur
        bytes_s = pkts_s * np.random.randint(300, 1200)
        fwd_mean = np.random.randint(300, 1200)
        bwd_mean = np.random.randint(20, 100)
        fin = np.random.randint(0, 5)
        syn = np.random.randint(100, 400)
        rst = np.random.randint(10, 80)
        
    row = {
        "flow_duration": dur,
        "fwd_packets": fwd,
        "bwd_packets": bwd,
        "flow_bytes_per_sec": bytes_s,
        "flow_pkts_per_sec": pkts_s,
        "fwd_pkt_len_mean": fwd_mean,
        "bwd_pkt_len_mean": bwd_mean,
        "fin_flag_cnt": fin,
        "syn_flag_cnt": syn,
        "rst_flag_cnt": rst,
    }
    
    # Apply aliases to test normalization
    if has_aliases:
        alias_map = {
            "flow_duration": "Duration",
            "flow_pkts_per_sec": "packets_per_sec",
            "flow_bytes_per_sec": "bytes_per_second",
            "fwd_packets": "fwd_packet_count",
            "bwd_packets": "bwd_packet_count",
            "fwd_pkt_len_mean": "fwd_packet_length_mean",
            "bwd_pkt_len_mean": "bwd_packet_length_mean",
            "fin_flag_cnt": "fin_flag_count",
            "syn_flag_cnt": "syn_flag_count",
            "rst_flag_cnt": "rst_flag_count",
        }
        row = {alias_map.get(k, k): v for k, v in row.items()}
        
    # Remove columns for testing missing/strict checks
    for col in missing_cols:
        row.pop(col, None)
        
    return row

def create_mock_dataset():
    """Create a suite of mock CSV files representing real N-BaIoT and simulated traffic."""
    print("[*] Creating temporary mock dataset directory...")
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    REAL_MOCK_DIR.mkdir(parents=True, exist_ok=True)
    SIM_MOCK_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Create simulated normal traffic
    rows_normal = [generate_mock_row(0, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_normal).to_csv(SIM_MOCK_DIR / "normal.csv", index=False)
    
    # 2. Create simulated DoS traffic with aliases
    rows_dos = [generate_mock_row(1, has_aliases=True) for _ in range(25)]
    pd.DataFrame(rows_dos).to_csv(SIM_MOCK_DIR / "dos.csv", index=False)
    
    # 3. Create simulated Port Scan traffic
    rows_scan = [generate_mock_row(2, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_scan).to_csv(SIM_MOCK_DIR / "port_scan.csv", index=False)
    
    # 4. Create simulated Mirai traffic
    rows_mirai = [generate_mock_row(3, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_mirai).to_csv(SIM_MOCK_DIR / "mirai.csv", index=False)
    
    # 5. Create real N-BaIoT benign data (under N-BaIoT/danmini_doorbell/benign_traffic.csv)
    doorbell_dir = REAL_MOCK_DIR / "danmini_doorbell"
    doorbell_dir.mkdir(parents=True, exist_ok=True)
    rows_real_benign = [generate_mock_row(0, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_real_benign).to_csv(doorbell_dir / "benign_traffic.csv", index=False)
    
    # 6. Create real N-BaIoT Mirai attack (under N-BaIoT/danmini_doorbell/mirai_attacks/scan.csv)
    mirai_dir = doorbell_dir / "mirai_attacks"
    mirai_dir.mkdir(parents=True, exist_ok=True)
    rows_real_mirai = [generate_mock_row(3, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_real_mirai).to_csv(mirai_dir / "scan.csv", index=False)
    
    # 7. Create real N-BaIoT DoS attack (under N-BaIoT/danmini_doorbell/gafgyt_attacks/udp.csv)
    gafgyt_dir = doorbell_dir / "gafgyt_attacks"
    gafgyt_dir.mkdir(parents=True, exist_ok=True)
    rows_real_dos = [generate_mock_row(1, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_real_dos).to_csv(gafgyt_dir / "udp.csv", index=False)
    
    # 8. Create real N-BaIoT Port Scan attack (under N-BaIoT/danmini_doorbell/gafgyt_attacks/scan.csv)
    # This matches gafgyt scan
    rows_real_scan = [generate_mock_row(2, has_aliases=False) for _ in range(25)]
    pd.DataFrame(rows_real_scan).to_csv(gafgyt_dir / "scan.csv", index=False)
    
    # 9. Create a bad CSV with missing features (to test warnings/strict mode)
    rows_bad = [generate_mock_row(0, has_aliases=False, missing_cols=["flow_duration", "rst_flag_cnt"]) for _ in range(5)]
    pd.DataFrame(rows_bad).to_csv(SIM_MOCK_DIR / "bad_incomplete.csv", index=False)
    
    print("[*] Mock datasets created successfully.")

def run_builder(args: list) -> subprocess.CompletedProcess:
    """Run dataset_builder.py with given arguments."""
    cmd = [sys.executable, str(BASE_DIR / "dataset_builder.py")] + args
    return subprocess.run(cmd, capture_output=True, text=True)

def verify_output_dataframe(path: Path):
    """Load and print validation details of output DataFrame."""
    print(f"\n--- Verifying output: {path.name} ---")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Label class counts:")
    print(df["label"].value_counts().sort_index())
    
    na_count = df.isna().sum().sum()
    print(f"Total NaN/Null values: {na_count}")
    
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    print(f"Total Infinite values: {inf_count}")
    
    # Assert checks
    assert df.shape[0] > 0, "DataFrame is empty!"
    assert na_count == 0, "Found NaN values in dataset!"
    assert inf_count == 0, "Found infinite values in dataset!"
    assert set(df["label"].unique()).issubset({0, 1, 2, 3}), "Labels contain values other than 0, 1, 2, 3"
    
    if path.name == "labelled.csv":
        expected_cols = REQUIRED_FEATURES + ["label"]
        assert df.columns.tolist() == expected_cols, f"Column mismatch! Expected: {expected_cols}, got: {df.columns.tolist()}"
    print(f"[/] Verification PASSED for {path.name}")

def run_tests():
    # Setup seed
    np.random.seed(42)
    
    # 1. Create mock data
    create_mock_dataset()
    
    try:
        # 2. Test 1: Syntax check & help output
        print("\n[*] Test 1: Compile and import check...")
        res = subprocess.run([sys.executable, "-m", "py_compile", str(BASE_DIR / "dataset_builder.py")], capture_output=True)
        if res.returncode != 0:
            print("[!] Syntax compilation failed!")
            print(res.stderr.decode())
            sys.exit(1)
        print("[/] Syntax check passed.")
        
        # 3. Test 2: Execute build without SMOTE (strict=False)
        print("\n[*] Test 2: Build combined dataset without SMOTE (strict mode off)...")
        # In non-strict mode, it should warn about bad_incomplete.csv and skip it, but successfully load the rest.
        args_no_smote = [
            "--real-data-dir", str(REAL_MOCK_DIR),
            "--sim-data-dir", str(SIM_MOCK_DIR),
            "--output", str(OUTPUT_FILE),
            "--audit-output", str(AUDIT_FILE),
            "--summary-output", str(SUMMARY_FILE),
            "--balance", "none"
        ]
        res = run_builder(args_no_smote)
        print("--- Console Output ---")
        print(res.stdout)
        if res.returncode != 0:
            print("[!] Builder failed to execute!")
            print(res.stderr)
            sys.exit(1)
            
        # Verify output files
        assert OUTPUT_FILE.exists(), f"Output file missing: {OUTPUT_FILE}"
        assert AUDIT_FILE.exists(), f"Audit file missing: {AUDIT_FILE}"
        assert SUMMARY_FILE.exists(), f"Summary JSON missing: {SUMMARY_FILE}"
        
        # Load and print stats
        verify_output_dataframe(OUTPUT_FILE)
        
        # Check summary JSON keys
        with open(SUMMARY_FILE, "r") as f:
            summary = json.load(f)
        required_keys = [
            "generation_time", "input_files", "loaded_files", "skipped_files",
            "unmapped_files", "rows_before_cleaning", "rows_after_cleaning",
            "duplicates_removed", "missing_rows_removed", "class_counts",
            "class_percentages", "source_counts", "imbalance_ratio",
            "smote_applied", "feature_order", "output_paths"
        ]
        for key in required_keys:
            assert key in summary, f"Summary JSON is missing required key: {key}"
        print("[/] Summary JSON keys verification passed.")
        
        # 4. Test 3: Execute build with strict=True (should crash on bad_incomplete.csv)
        print("\n[*] Test 3: Build combined dataset with strict mode ON (expecting crash due to incomplete file)...")
        args_strict = args_no_smote + ["--strict"]
        res = run_builder(args_strict)
        if res.returncode == 0:
            print("[!] ERROR: Strict mode did not halt execution on incomplete file!")
            sys.exit(1)
        else:
            print(f"[/] Strict mode successfully halted execution. Error output:\n{res.stderr.strip()}")
            
        # 5. Test 4: Execute build with SMOTE
        print("\n[*] Test 4: Build combined dataset with SMOTE enabled...")
        args_smote = [
            "--real-data-dir", str(REAL_MOCK_DIR),
            "--sim-data-dir", str(SIM_MOCK_DIR),
            "--output", str(OUTPUT_FILE),
            "--audit-output", str(AUDIT_FILE),
            "--summary-output", str(SUMMARY_FILE),
            "--balance", "smote",
            "--test-size", "0.20",
            "--random-state", "42"
        ]
        res = run_builder(args_smote)
        print("--- Console Output ---")
        print(res.stdout)
        if res.returncode != 0:
            print("[!] Builder failed under SMOTE execution!")
            print(res.stderr)
            sys.exit(1)
            
        train_orig_path = MOCK_DIR / "train_original.csv"
        train_smote_path = MOCK_DIR / "train_smote.csv"
        test_path = MOCK_DIR / "test.csv"
        
        assert train_orig_path.exists(), "train_original.csv is missing!"
        assert train_smote_path.exists(), "train_smote.csv is missing!"
        assert test_path.exists(), "test.csv is missing!"
        
        verify_output_dataframe(train_orig_path)
        verify_output_dataframe(train_smote_path)
        verify_output_dataframe(test_path)
        
        # Check SMOTE balance: class counts in train_smote should be perfectly equal
        df_smote = pd.read_csv(train_smote_path)
        counts = df_smote["label"].value_counts()
        print("\nResampled training class counts:")
        print(counts)
        assert len(counts.unique()) == 1, "SMOTE training set classes are not balanced!"
        print("[/] SMOTE balance verification passed.")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        
    finally:
        # Cleanup temporary mock files
        print("\n[*] Cleaning up temporary mock directory...")
        if MOCK_DIR.exists():
            import gc
            import stat
            gc.collect()  # Force close any remaining file handles
            
            def remove_readonly(func, path, excinfo):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            
            shutil.rmtree(MOCK_DIR, onerror=remove_readonly)
        print("[/] Cleanup complete.")

if __name__ == "__main__":
    run_tests()
