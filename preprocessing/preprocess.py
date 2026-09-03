"""
Data Preprocessing Engine for CyberWorld-AI.
Loads raw CSV files from data/raw/, normalizes schema, cleans missing/infinite values,
encodes binary Attack label and MITRE stage labels, and saves processed tabular data.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
import yaml

from preprocessing.check_dataset import detect_column, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard stage mapping from attack string names
STAGE_MAPPING_RULES = {
    "benign": 0,
    "normal": 0,
    "portscan": 1,
    "reconnaissance": 1,
    "ssh-bruteforce": 2,
    "ftp-bruteforce": 2,
    "brute force": 2,
    "dos": 3,
    "ddos": 3,
    "bot": 4,
    "c2": 4,
    "command and control": 4,
    "infilteration": 4,
    "web attack": 2,
    "exfiltration": 5,
    "data-exfiltration": 5
}

def map_attack_stage(label_str):
    """Map string attack label to numerical MITRE ATT&CK stage index (0..5)."""
    if not isinstance(label_str, str):
        return 0
    clean_label = label_str.strip().lower()
    for pattern, stage in STAGE_MAPPING_RULES.items():
        if pattern in clean_label:
            return stage
    return 1 if clean_label != "benign" and clean_label != "normal" else 0

def clean_dataframe(df, drop_zero_variance=True):
    """Clean column names, handle NaNs, infs, duplicate rows, and data types."""
    # 1. Clean column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. Drop duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates().copy()
    dropped_dups = initial_count - len(df)
    if dropped_dups > 0:
        logger.info(f"Dropped {dropped_dups} duplicate rows.")
        
    # 3. Detect key columns
    label_col = detect_column(df.columns, "label")
    timestamp_col = detect_column(df.columns, "timestamp")
    
    if not label_col:
        raise ValueError("Could not locate a valid Label column in dataset.")
        
    # 4. Standardize binary Attack label and Attack Stage label
    df["Original_Label"] = df[label_col].astype(str)
    df["Attack"] = df["Original_Label"].apply(lambda x: 0 if x.strip().upper() in ["BENIGN", "NORMAL", "0"] else 1)
    df["Attack_Stage"] = df["Original_Label"].apply(map_attack_stage)
    
    # 5. Standardize Timestamp column
    if timestamp_col:
        df["Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        # Drop rows with invalid timestamps
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    else:
        logger.warning("Timestamp column not found. Generating synthetic timestamps.")
        df["Timestamp"] = [pd.Timestamp("2026-08-31 10:00:00") + pd.Timedelta(seconds=i * 0.5) for i in range(len(df))]
        
    # 6. Drop unusable non-numeric metadata columns (except Timestamp, Original_Label, Attack, Attack_Stage)
    reserved_cols = ["Timestamp", "Original_Label", "Attack", "Attack_Stage"]
    
    numeric_df_cols = []
    for col in df.columns:
        if col in reserved_cols or col == label_col or col == timestamp_col:
            continue
        # Try converting column to float
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_df_cols.append(col)
        
    # Replace inf and -inf with NaN
    df[numeric_df_cols] = df[numeric_df_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column median
    df[numeric_df_cols] = df[numeric_df_cols].fillna(df[numeric_df_cols].median())
    
    # Fill remaining NaNs (if any median was NaN) with 0
    df[numeric_df_cols] = df[numeric_df_cols].fillna(0)
    
    # Drop zero variance (constant) numeric columns (only during training preprocessing)
    if drop_zero_variance:
        std = df[numeric_df_cols].std()
        constant_cols = std[std == 0].index.tolist()
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} zero-variance numeric columns.")
            df = df.drop(columns=constant_cols)
            numeric_df_cols = [c for c in numeric_df_cols if c not in constant_cols]

    logger.info(f"Cleaned dataset: {len(df)} rows, {len(numeric_df_cols)} numeric feature columns.")
    return df

def preprocess_all(config_path="config.yaml"):
    """Load raw CSV files, clean them, and save processed CSV."""
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_data_dir"])
    output_path = Path(config["paths"]["processed_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        from preprocessing.check_dataset import check_dataset
        logger.info("No CSV raw files found. Executing check_dataset auto-generation...")
        check_dataset(config_path)
        csv_files = list(raw_dir.glob("*.csv"))
        
    processed_dfs = []
    for file_path in csv_files:
        logger.info(f"Processing raw file: {file_path.name}")
        df_raw = pd.read_csv(file_path, low_memory=False)
        df_clean = clean_dataframe(df_raw)
        processed_dfs.append(df_clean)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Ensure chronological order
    if "Timestamp" in final_df.columns:
        final_df = final_df.sort_values("Timestamp").reset_index(drop=True)
        
    final_df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed dataset saved successfully to {output_path.resolve()}")
    logger.info(f"Summary: {len(final_df)} rows, {len(final_df.columns)} columns.")
    logger.info(f"Binary Attack Class Distribution: {final_df['Attack'].value_counts().to_dict()}")
    logger.info(f"Attack Stage Distribution: {final_df['Attack_Stage'].value_counts().to_dict()}")
    return output_path

if __name__ == "__main__":
    preprocess_all()
