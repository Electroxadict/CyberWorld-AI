"""
Temporal Windowing and Behavioral Feature Extractor for CyberWorld-AI.
Aggregates clean flow data into fixed time windows (default 5s) and computes
both flow-level statistical features and macro behavioral features.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
import yaml

from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of reserved meta columns that are NOT feature inputs
RESERVED_COLUMNS = ["Timestamp", "Original_Label", "Attack", "Attack_Stage", "Window_ID"]

def aggregate_time_windows(processed_csv_path=None, config_path="config.yaml"):
    """
    Groups flow records into fixed temporal windows and extracts flow + behavioral features.
    
    Returns:
        pd.DataFrame: Aggregated time window feature matrix with targets.
    """
    config = load_config(config_path)
    window_sec = config.get("temporal", {}).get("time_window_seconds", 5)
    
    if processed_csv_path is None:
        processed_csv_path = Path(config["paths"]["processed_data"])
    else:
        processed_csv_path = Path(processed_csv_path)
        
    if not processed_csv_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_csv_path}. Run preprocess.py first.")
        
    logger.info(f"Loading preprocessed traffic data from {processed_csv_path}...")
    df = pd.read_csv(processed_csv_path)
    
    # 1. Handle Timestamps & Chronological Sorting
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
    else:
        logger.warning("No Timestamp column found. Creating synthetic 0.5s timestamps.")
        base_t = pd.Timestamp("2026-08-31 10:00:00")
        df["Timestamp"] = [base_t + pd.Timedelta(seconds=i * 0.5) for i in range(len(df))]
        
    # 2. Assign Time Window Group ID based on timestamp interval
    min_time = df["Timestamp"].min()
    df["Window_ID"] = ((df["Timestamp"] - min_time).dt.total_seconds() // window_sec).astype(int)
    
    grouped = df.groupby("Window_ID")
    logger.info(f"Grouping {len(df)} traffic rows into {grouped.ngroups} temporal windows ({window_sec}s each)...")
    
    window_records = []
    
    # Pre-identify numeric feature columns in raw flow dataframe
    flow_num_cols = [c for c in df.columns if c not in RESERVED_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    
    # Moving window traffic history tracker for burstiness calculation
    window_pkts_history = []
    
    for window_id, group in grouped:
        record = {}
        record["Window_ID"] = window_id
        record["Window_Timestamp"] = group["Timestamp"].min()
        record["Flow_Count"] = len(group)
        
        # --- Aggregated Flow Statistics ---
        for col in flow_num_cols:
            vals = group[col]
            record[f"{col}_mean"] = vals.mean()
            record[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0
            record[f"{col}_max"] = vals.max()
            record[f"{col}_min"] = vals.min()
            
        # Fill any NaNs created by std dev of single element
        for k in list(record.keys()):
            if isinstance(record[k], float) and np.isnan(record[k]):
                record[k] = 0.0
                
        # --- Macro Behavioral Indicators ---
        src_port_col = [c for c in group.columns if "src" in c.lower() and "port" in c.lower()]
        dst_port_col = [c for c in group.columns if "dst" in c.lower() and "port" in c.lower()]
        
        unique_src_ports = group[src_port_col[0]].nunique() if src_port_col else 1
        unique_dst_ports = group[dst_port_col[0]].nunique() if dst_port_col else 1
        
        # Flag Counts across window if present
        syn_cnt = group["SYN Flag Cnt"].sum() if "SYN Flag Cnt" in group.columns else 0
        ack_cnt = group["ACK Flag Cnt"].sum() if "ACK Flag Cnt" in group.columns else 0
        rst_cnt = group["RST Flag Cnt"].sum() if "RST Flag Cnt" in group.columns else 0
        
        tot_pkts = (group["Tot Fwd Pkts"].sum() + group["Tot Bwd Pkts"].sum()) if "Tot Fwd Pkts" in group.columns and "Tot Bwd Pkts" in group.columns else len(group)
        tot_bytes = (group["TotLen Fwd Pkts"].sum() + group["TotLen Bwd Pkts"].sum()) if "TotLen Fwd Pkts" in group.columns and "TotLen Bwd Pkts" in group.columns else 0
        
        record["Unique_Src_Ports"] = unique_src_ports
        record["Unique_Dst_Ports"] = unique_dst_ports
        record["SYN_ACK_Ratio"] = float(syn_cnt) / (float(ack_cnt) + 1.0)
        record["RST_SYN_Ratio"] = float(rst_cnt) / (float(syn_cnt) + 1.0)
        record["Port_Diversity"] = float(unique_dst_ports) / (float(len(group)) + 1.0)
        record["Connection_Failure_Rate"] = float(rst_cnt) / (float(tot_pkts) + 1.0)
        
        # Traffic Burstiness Score relative to moving mean
        window_pkts_history.append(tot_pkts)
        moving_mean_pkts = np.mean(window_pkts_history[-5:]) # past 5 windows
        record["Traffic_Burst_Score"] = float(tot_pkts) / (float(moving_mean_pkts) + 1.0)
        record["Port_Scan_Score"] = float(unique_dst_ports * syn_cnt) / (float(len(group)) + 1.0)
        
        # --- Targets for Window ---
        # Attack = 1 if ANY flow in window is attack
        record["Attack"] = 1 if (group["Attack"] == 1).any() else 0
        # Stage = Maximum attack stage observed in window
        record["Attack_Stage"] = int(group["Attack_Stage"].max()) if "Attack_Stage" in group.columns else 0
        
        window_records.append(record)
        
    window_df = pd.DataFrame(window_records)
    
    # Sort chronologically by Window_ID
    window_df = window_df.sort_values("Window_ID").reset_index(drop=True)
    
    # Output path
    output_dir = Path(config["paths"]["processed_data"]).parent
    output_csv = output_dir / "time_windows.csv"
    output_npy = output_dir / "time_windows.npy"
    
    window_df.to_csv(output_csv, index=False)
    
    # Extract purely numeric feature columns for numpy state sequence saving
    feature_cols = [c for c in window_df.columns if c not in ["Window_ID", "Window_Timestamp", "Attack", "Attack_Stage"]]
    feature_matrix = window_df[feature_cols].values.astype(np.float32)
    np.save(output_npy, feature_matrix)
    
    logger.info(f"Generated {len(window_df)} temporal state windows.")
    logger.info(f"Feature dimension per window state: {len(feature_cols)} features.")
    logger.info(f"Saved aggregated windows to {output_csv} and {output_npy}")
    
    return window_df

if __name__ == "__main__":
    aggregate_time_windows()
