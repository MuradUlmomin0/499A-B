#!/usr/bin/env python3
"""
PRIVATE SHIELD - Week 3 Person B Dataset Builder
Privacy-Preserving Federated Intelligence System for IoT Threat Detection
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dataset_builder")

# Global Config
RANDOM_STATE = 42

# Label mapping keywords and priorities
LABEL_KEYWORDS = {
    "mirai": 3,
    "port_scan": 2,
    "scan": 2,
    "dos": 1,
    "flood": 1,
    "udp": 1,
    "tcp": 1,
    "combo": 1,
    "junk": 1,
    "normal": 0,
    "benign": 0
}

LABEL_PRIORITY = ["mirai", "port_scan", "scan", "dos", "flood", "udp", "tcp", "combo", "junk", "normal", "benign"]

LABEL_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Port Scan",
    3: "Mirai"
}

# Column aliases dictionary to map various naming formats
COLUMN_ALIASES = {
    "duration": "flow_duration",
    "packets_per_sec": "flow_pkts_per_sec",
    "bytes_per_second": "flow_bytes_per_sec",
    "fwd_packet_count": "fwd_packets",
    "bwd_packet_count": "bwd_packets",
    "fwd_packet_length_mean": "fwd_pkt_len_mean",
    "bwd_packet_length_mean": "bwd_pkt_len_mean",
    "fin_flag_count": "fin_flag_cnt",
    "syn_flag_count": "syn_flag_cnt",
    "rst_flag_count": "rst_flag_cnt",
    
    # Common variations
    "flow duration": "flow_duration",
    "flow-duration": "flow_duration",
    "tot fwd pkts": "fwd_packets",
    "tot bwd pkts": "bwd_packets",
    "tot_fwd_pkts": "fwd_packets",
    "tot_bwd_pkts": "bwd_packets",
    "flow bytes/s": "flow_bytes_per_sec",
    "flow pkts/s": "flow_pkts_per_sec",
    "fwd pkt len mean": "fwd_pkt_len_mean",
    "bwd pkt len mean": "bwd_pkt_len_mean",
    "fin flag cnt": "fin_flag_cnt",
    "syn flag cnt": "syn_flag_cnt",
    "rst flag cnt": "rst_flag_cnt",
}

# Machine learning feature columns in exact order
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

def normalize_column_name(col: str) -> str:
    """Convert column name to lowercase snake_case and resolve configured aliases."""
    if not isinstance(col, str):
        return str(col)
    clean = col.strip().lower()
    clean = clean.replace(" ", "_").replace("-", "_").replace("/", "_")
    if clean in COLUMN_ALIASES:
        return COLUMN_ALIASES[clean]
    return clean

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names."""
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df

def discover_csv_files(directory: Path) -> List[Path]:
    """Find all CSV files in the given directory recursively."""
    if not directory.exists():
        return []
    return sorted(list(directory.rglob("*.csv")))

def infer_label_from_path(file_path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Infers label and matching keyword from file path or filename."""
    path_str = str(file_path).lower().replace("\\", "/")
    for kw in LABEL_PRIORITY:
        if kw in path_str:
            return LABEL_KEYWORDS[kw], kw
    return None, None

def infer_label(df: pd.DataFrame, file_path: Path) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    """
    Infers the label for rows in df.
    Checks:
    1. If a 'label' or similar column exists and maps.
    2. Path/filename keywords.
    
    Returns (labels_series, detected_class, success)
    """
    label_cols = [c for c in df.columns if c.lower() in ["label", "class", "attack_type", "attack", "mode", "traffic_type"]]
    
    if label_cols:
        col = label_cols[0]
        unique_vals = df[col].dropna().unique()
        
        # If numeric and values are valid labels (0-3)
        is_numeric = all(isinstance(v, (int, float, np.integer)) and int(v) in [0, 1, 2, 3] for v in unique_vals)
        if is_numeric:
            labels = df[col].astype(int)
            most_common = labels.mode()[0] if not labels.empty else 0
            return labels, LABEL_NAMES.get(most_common, "unknown"), True
            
        # Try mapping text labels in the column
        mapped_labels = []
        for val in df[col]:
            val_str = str(val).lower()
            mapped_val = None
            for kw in LABEL_PRIORITY:
                if kw in val_str:
                    mapped_val = LABEL_KEYWORDS[kw]
                    break
            mapped_labels.append(mapped_val)
            
        mapped_series = pd.Series(mapped_labels)
        if mapped_series.isna().sum() == 0:
            most_common_mapped = mapped_series.mode()[0] if not mapped_series.empty else 0
            return mapped_series.astype(int), LABEL_NAMES.get(most_common_mapped, "unknown"), True

    # Fallback to path inference
    label_val, keyword = infer_label_from_path(file_path)
    if label_val is not None:
        labels = pd.Series([label_val] * len(df))
        return labels, keyword, True
        
    return None, None, False

def validate_required_features(df: pd.DataFrame, filename: str, strict: bool) -> Tuple[bool, List[str]]:
    """Check if the DataFrame contains all required feature columns."""
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        print(f"\n[!] WARNING: Missing required features in file: {filename}")
        print(f"    Missing features: {missing}")
        print(f"    Available columns: {list(df.columns)}")
        print("    Which teammate/script must generate these: Person A's Week 3 feature-generation update.")
        
        if strict:
            print("    [!] Strict mode is active. Stopping execution.")
            sys.exit(f"Error: Missing required features {missing} in file {filename}")
        else:
            print("    [!] Strict mode is inactive. Skipping file.\n")
            return False, missing
    return True, []

def derive_supported_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features if valid. E.g. flow_bytes_per_sec = flow_pkts_per_sec * mean_packet_length"""
    df = df.copy()
    if "flow_bytes_per_sec" not in df.columns or df["flow_bytes_per_sec"].isna().sum() > 0:
        if "flow_pkts_per_sec" in df.columns:
            mean_len_col = None
            for col in ["fwd_pkt_len_mean", "bytes_per_pkt", "mean_packet_length"]:
                if col in df.columns:
                    mean_len_col = col
                    break
            if mean_len_col is not None:
                calc_val = df["flow_pkts_per_sec"] * df[mean_len_col]
                if "flow_bytes_per_sec" not in df.columns:
                    df["flow_bytes_per_sec"] = calc_val
                else:
                    df["flow_bytes_per_sec"] = df["flow_bytes_per_sec"].fillna(calc_val)
    return df

def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean the DataFrame and return cleaning stats.
    Replaces infinities, drops missing features/duplicates/invalid labels/negative metrics.
    """
    stats = {
        "initial_rows": len(df),
        "infinities_replaced": 0,
        "missing_features_removed": 0,
        "duplicates_removed": 0,
        "invalid_labels_removed": 0,
        "negative_values_removed": 0
    }
    
    df = df.copy()
    
    # 1. Infinities check
    for col in REQUIRED_FEATURES:
        if col in df.columns:
            inf_mask = df[col].isin([np.inf, -np.inf])
            stats["infinities_replaced"] += int(inf_mask.sum())
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors='coerce')
        
    # 2. Reject invalid labels
    if "label" in df.columns:
        invalid_mask = df["label"].isna() | (~df["label"].isin([0, 1, 2, 3]))
        stats["invalid_labels_removed"] = int(invalid_mask.sum())
        df = df[~invalid_mask]
        df["label"] = df["label"].astype(int)
        
    # 3. Remove rows with missing required features or labels
    cols_to_check = [c for c in REQUIRED_FEATURES if c in df.columns] + ["label"]
    missing_mask = df[cols_to_check].isna().any(axis=1)
    stats["missing_features_removed"] = int(missing_mask.sum())
    df = df[~missing_mask]
    
    # 4. Reject negative values in fields where negative values are impossible
    negative_mask = (df[REQUIRED_FEATURES] < 0).any(axis=1)
    stats["negative_values_removed"] = int(negative_mask.sum())
    df = df[~negative_mask]
    
    # 5. Remove exact duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = int(before_dup - len(df))
    
    return df, stats

def load_csv_safely(file_path: Path, max_rows: Optional[int], random_state: int) -> Optional[pd.DataFrame]:
    """Load a CSV file safely handling encoding and empty files."""
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.warning(f"File is empty or missing: {file_path.name}")
            return None
        
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")
            
        if df.empty:
            logger.warning(f"No rows found in file: {file_path.name}")
            return None
            
        # Reproducible sampling if max_rows is specified
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state)
            
        return df
    except Exception as e:
        logger.error(f"Failed to load file {file_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="PRIVATE SHIELD Week 3 Dataset Builder - Combines & prepares dataset for IoT threat detection"
    )
    parser.add_argument("--real-data-dir", type=str, default="data/N-BaIoT", help="Directory of real N-BaIoT dataset")
    parser.add_argument("--sim-data-dir", type=str, default="data", help="Directory of locally simulated dataset")
    parser.add_argument("--output", type=str, default="data/labelled.csv", help="Output path for combined dataset")
    parser.add_argument("--audit-output", type=str, default="data/labelled_audit.csv", help="Output path for audit dataset")
    parser.add_argument("--summary-output", type=str, default="data/dataset_summary.json", help="Output path for JSON summary")
    parser.add_argument("--balance", type=str, choices=["none", "smote"], default="none", help="Balancing mode (none or smote)")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set size ratio for train-test split")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random state seed")
    parser.add_argument("--strict", action="store_true", help="Crash on any missing features or failed mappings")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Max number of rows to sample per file")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    real_dir = Path(args.real_data_dir)
    sim_dir = Path(args.sim_data_dir)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    summary_path = Path(args.summary_output)
        
    print("============================================================")
    print("PRIVATE SHIELD - WEEK 3 DATASET BUILDER")
    print("============================================================")
    
    # 1. Discover files
    real_files = discover_csv_files(real_dir)
    # Simulated files are in sim_dir directly or subdirectories, excluding N-BaIoT dir
    all_sim_files = discover_csv_files(sim_dir)
    sim_files = [f for f in all_sim_files if not str(f.resolve()).startswith(str(real_dir.resolve()))]
    # Filter out output files if they are in the same folder to prevent circular loading
    output_names = [output_path.name, audit_path.name, "train_original.csv", "train_smote.csv", "test.csv"]
    sim_files = [f for f in sim_files if f.name not in output_names]
    
    print(f"Real dataset files found: {len(real_files)}")
    print(f"Simulated dataset files found: {len(sim_files)}")
    
    loaded_dfs = []
    summary_input_files = []
    loaded_files_list = []
    skipped_files_list = []
    unmapped_files_list = []
    
    total_initial_rows = 0
    total_duplicates_removed = 0
    total_missing_rows_removed = 0
    total_invalid_labels_removed = 0
    total_negative_removed = 0
    total_inf_replaced = 0
    
    # Process Real Files
    for file_path in real_files:
        summary_input_files.append(str(file_path))
        df = load_csv_safely(file_path, args.max_rows_per_file, args.random_state)
        if df is None:
            skipped_files_list.append(str(file_path))
            continue
            
        df = standardize_columns(df)
        labels_series, detected_class, success = infer_label(df, file_path)
        if not success:
            print(f"[!] Unmapped file skipped: {file_path.name} (Cannot infer label)")
            unmapped_files_list.append(str(file_path))
            if args.strict:
                sys.exit(f"Error: Could not confidently map class for file {file_path}")
            continue
            
        df["label"] = labels_series
        df = derive_supported_features(df)
        
        is_valid, _ = validate_required_features(df, file_path.name, args.strict)
        if not is_valid:
            skipped_files_list.append(str(file_path))
            continue
            
        # Clean this individual dataframe
        df, stats = clean_dataframe(df)
        
        # Accumulate stats
        total_initial_rows += stats["initial_rows"]
        total_inf_replaced += stats["infinities_replaced"]
        total_missing_rows_removed += stats["missing_features_removed"]
        total_duplicates_removed += stats["duplicates_removed"]
        total_invalid_labels_removed += stats["invalid_labels_removed"]
        total_negative_removed += stats["negative_values_removed"]
        
        # Set source and metadata
        df["source"] = "real_nbaiot"
        df["filename"] = file_path.name
        df["attack_name"] = detected_class
        
        loaded_dfs.append(df)
        loaded_files_list.append(str(file_path))
        print(f"Loaded: {file_path.name} | Class: {detected_class} | Label: {df['label'].iloc[0] if len(df) > 0 else 'N/A'} | Rows: {len(df)}")

    # Process Simulated Files
    for file_path in sim_files:
        summary_input_files.append(str(file_path))
        df = load_csv_safely(file_path, args.max_rows_per_file, args.random_state)
        if df is None:
            skipped_files_list.append(str(file_path))
            continue
            
        df = standardize_columns(df)
        labels_series, detected_class, success = infer_label(df, file_path)
        if not success:
            print(f"[!] Unmapped simulated file skipped: {file_path.name}")
            unmapped_files_list.append(str(file_path))
            if args.strict:
                sys.exit(f"Error: Could not confidently map class for file {file_path}")
            continue
            
        df["label"] = labels_series
        df = derive_supported_features(df)
        
        is_valid, _ = validate_required_features(df, file_path.name, args.strict)
        if not is_valid:
            skipped_files_list.append(str(file_path))
            continue
            
        df, stats = clean_dataframe(df)
        
        total_initial_rows += stats["initial_rows"]
        total_inf_replaced += stats["infinities_replaced"]
        total_missing_rows_removed += stats["missing_features_removed"]
        total_duplicates_removed += stats["duplicates_removed"]
        total_invalid_labels_removed += stats["invalid_labels_removed"]
        total_negative_removed += stats["negative_values_removed"]
        
        # Assign simulated sources
        df["source"] = df.apply(lambda r: f"simulated_{LABEL_NAMES.get(r['label'], 'unknown').lower().replace(' ', '_')}", axis=1)
        df["filename"] = file_path.name
        df["attack_name"] = detected_class
        
        loaded_dfs.append(df)
        loaded_files_list.append(str(file_path))
        print(f"Loaded: {file_path.name} | Class: {detected_class} | Label: {df['label'].iloc[0] if len(df) > 0 else 'N/A'} | Rows: {len(df)}")
        
    print(f"Successfully loaded files: {len(loaded_files_list)}")
    print(f"Skipped files: {len(skipped_files_list)}")
    
    if not loaded_dfs:
        print("[!] ERROR: No dataframes loaded successfully. Cannot build dataset.")
        sys.exit(1)
        
    # Combine datasets
    combined_df = pd.concat(loaded_dfs, ignore_index=True)
    
    # Audit copy preserves everything
    audit_cols = ["source", "filename", "timestamp", "topic", "device_id", "attack_name"] + REQUIRED_FEATURES + ["label"]
    available_audit_cols = [c for c in audit_cols if c in combined_df.columns]
    audit_df = combined_df[available_audit_cols]
    
    # Final ML-only dataset contains ONLY the REQUIRED_FEATURES and label
    ml_df = combined_df[REQUIRED_FEATURES + ["label"]].copy()
    
    # Reject invalid labels or NaNs that might have slipped through concatenation
    ml_df = ml_df.dropna()
    ml_df["label"] = ml_df["label"].astype(int)
    
    print(f"\nRows before cleaning: {total_initial_rows}")
    print(f"Duplicate rows removed: {total_duplicates_removed}")
    print(f"Rows with invalid values removed: {total_missing_rows_removed + total_invalid_labels_removed + total_negative_removed}")
    print(f"Final rows: {len(ml_df)}")
    
    # Class Distribution calculations
    class_counts = ml_df["label"].value_counts().sort_index()
    total_samples = len(ml_df)
    class_percentages = {int(k): float(v / total_samples) for k, v in class_counts.items()}
    
    print("\nClass distribution:")
    for label, count in class_counts.items():
        pct = class_percentages[int(label)] * 100
        print(f"{label} - {LABEL_NAMES.get(label)}: {count} ({pct:.2f}%)")
        
    smallest_class_count = class_counts.min() if not class_counts.empty else 0
    largest_class_count = class_counts.max() if not class_counts.empty else 0
    imbalance_ratio = largest_class_count / smallest_class_count if smallest_class_count > 0 else float('inf')
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    
    is_imbalanced = imbalance_ratio > 2.0
    if is_imbalanced:
        print("[!] WARNING: The dataset is heavily imbalanced (ratio > 2.0).")
        
    # Source vs Label check
    print("\nSource vs Label Distribution Matrix:")
    crosstab_res = pd.crosstab(combined_df["source"], combined_df["label"])
    print(crosstab_res)
    
    # Check for leakage
    has_leakage = False
    for label in sorted(ml_df["label"].unique()):
        sources_with_label = combined_df[combined_df["label"] == label]["source"].unique()
        if len(sources_with_label) == 1:
            print(f"[!] WARNING: Label {label} ({LABEL_NAMES.get(label)}) exists ONLY in source: '{sources_with_label[0]}'")
            has_leakage = True
            
    if has_leakage:
        print("[!] CAUTION: Model could learn to classify the source instead of the attack pattern!")
        
    # Make sure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Main Cleaned Dataset
    ml_df.to_csv(output_path, index=False)
    print(f"\nSaved naturally distributed combined dataset to: {output_path}")
    
    # Save Audit Dataset
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved audit dataset to: {audit_path}")
    
    smote_applied = "No"
    
    # Handle Optional SMOTE
    if args.balance == "smote":
        print("\n--- SMOTE Balancing Process ---")
        if smallest_class_count < 2:
            print("[!] SMOTE skipped: Smallest class count must be at least 2.")
            smote_applied = "Skipped (Smallest class < 2)"
        else:
            try:
                from imblearn.over_sampling import SMOTE
                from sklearn.model_selection import train_test_split
                
                # Split first to prevent data leakage (Stratified split)
                X = ml_df[REQUIRED_FEATURES]
                y = ml_df["label"]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=args.test_size,
                    random_state=args.random_state,
                    stratify=y
                )
                
                train_original = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)
                
                # Save Train Original and Test Untouched
                train_orig_path = output_path.parent / "train_original.csv"
                test_path = output_path.parent / "test.csv"
                train_original.to_csv(train_orig_path, index=False)
                test_df.to_csv(test_path, index=False)
                
                print(f"Saved original training set to: {train_orig_path} (Rows: {len(train_original)})")
                print(f"Saved untouched testing set to: {test_path} (Rows: {len(test_df)})")
                
                # Apply SMOTE to training set only
                k_neighbors = min(5, smallest_class_count - 1)
                print(f"Configuring SMOTE with k_neighbors={k_neighbors}")
                smote = SMOTE(k_neighbors=k_neighbors, random_state=args.random_state)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                
                train_smote = pd.concat([X_train_res, y_train_res], axis=1)
                train_smote_path = output_path.parent / "train_smote.csv"
                train_smote.to_csv(train_smote_path, index=False)
                print(f"Saved SMOTE balanced training set to: {train_smote_path} (Rows: {len(train_smote)})")
                
                # Print balanced class counts
                print("\nBalanced training set class distribution:")
                res_counts = y_train_res.value_counts().sort_index()
                for label, count in res_counts.items():
                    print(f"{label} - {LABEL_NAMES.get(label)}: {count}")
                    
                smote_applied = "Yes"
                
            except ImportError:
                print("[!] ERROR: imbalanced-learn is not installed. Run 'pip install imbalanced-learn' to use SMOTE.")
                smote_applied = "Error (imbalanced-learn not installed)"
                if args.strict:
                    sys.exit("Error: Failed to import imbalanced-learn under strict mode.")
                    
    # Write JSON Summary
    summary_data = {
        "generation_time": datetime.now().isoformat(),
        "input_files": summary_input_files,
        "loaded_files": loaded_files_list,
        "skipped_files": skipped_files_list,
        "unmapped_files": unmapped_files_list,
        "rows_before_cleaning": total_initial_rows,
        "rows_after_cleaning": len(ml_df),
        "duplicates_removed": total_duplicates_removed,
        "missing_rows_removed": total_missing_rows_removed,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "class_percentages": class_percentages,
        "source_counts": {str(k): int(v) for k, v in combined_df["source"].value_counts().items()},
        "imbalance_ratio": float(imbalance_ratio) if not np.isinf(imbalance_ratio) else "inf",
        "smote_applied": smote_applied,
        "feature_order": REQUIRED_FEATURES,
        "output_paths": {
            "labelled": str(output_path),
            "labelled_audit": str(audit_path),
            "summary": str(summary_path),
            "train_original": str(output_path.parent / "train_original.csv") if smote_applied == "Yes" else None,
            "train_smote": str(output_path.parent / "train_smote.csv") if smote_applied == "Yes" else None,
            "test": str(output_path.parent / "test.csv") if smote_applied == "Yes" else None
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved summary stats to: {summary_path}")
    
    # 10. Print final descriptive stats for feature columns
    print("\nDescriptive statistics for final feature columns:")
    pd.set_option('display.max_columns', 5)
    print(ml_df[REQUIRED_FEATURES].describe())
    
    print("\n" + "="*60)
    print("Dataset build completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
