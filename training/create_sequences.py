"""
Temporal Sequence Generator for CyberWorld-AI.
Converts aggregated time windows into multi-step temporal sequences:
(S[t-9], S[t-8], ..., S[t]) -> predict S[t+1], Attack[t+1], Stage[t+1].
Maintains strict chronological sequence ordering and zero-leakage splits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd

from preprocessing.check_dataset import load_config
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.time_window import aggregate_time_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_temporal_sequences(data_csv_path=None, config_path="config.yaml"):
    """
    Creates temporal sequence windows for LSTM World Model training.
    
    Returns:
        dict: Dictionary containing train/val/test sequence arrays and targets.
    """
    config = load_config(config_path)
    seq_len = config.get("temporal", {}).get("sequence_length", 10)
    output_dir = Path(config["paths"]["sequences_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load time windows dataframe
    time_windows_csv = Path(config["paths"]["processed_data"]).parent / "time_windows.csv"
    if not time_windows_csv.exists():
        logger.info("time_windows.csv not found. Aggregating time windows...")
        df = aggregate_time_windows(config_path=config_path)
    else:
        df = pd.read_csv(time_windows_csv)
        
    logger.info(f"Loaded {len(df)} total time windows. Generating sequences of length {seq_len}...")
    
    reserved = ["Window_ID", "Window_Timestamp", "Attack", "Attack_Stage"]
    feature_cols = [c for c in df.columns if c not in reserved]
    
    features = df[feature_cols].values.astype(np.float32)
    attacks = df["Attack"].values.astype(np.int64)
    stages = df["Attack_Stage"].values.astype(np.int64)
    
    total_samples = len(df)
    if total_samples <= seq_len:
        raise ValueError(f"Dataset length ({total_samples}) must be greater than sequence length ({seq_len}).")
        
    # --- Generate Sliding Window Sequences (No Shuffling) ---
    X_seq_list = []
    Y_state_list = []
    Y_attack_list = []
    Y_stage_list = []
    
    for i in range(total_samples - seq_len):
        X_seq_list.append(features[i : i + seq_len])
        Y_state_list.append(features[i + seq_len])
        Y_attack_list.append(attacks[i + seq_len])
        Y_stage_list.append(stages[i + seq_len])
        
    X_seq = np.array(X_seq_list, dtype=np.float32)      # Shape: (N, seq_len, num_features)
    Y_state = np.array(Y_state_list, dtype=np.float32)  # Shape: (N, num_features)
    Y_attack = np.array(Y_attack_list, dtype=np.int64)  # Shape: (N,)
    Y_stage = np.array(Y_stage_list, dtype=np.int64)    # Shape: (N,)
    
    logger.info(f"Created {len(X_seq)} total sequences of shape {X_seq.shape}")
    
    # --- Chronological Dataset Splitting (60% Train, 20% Val, 20% Test) ---
    num_seq = len(X_seq)
    train_end = int(num_seq * config["training"].get("train_split", 0.60))
    val_end = int(num_seq * (config["training"].get("train_split", 0.60) + config["training"].get("val_split", 0.20)))
    
    # Slice sequence splits
    X_train_raw = X_seq[:train_end]
    Y_train_state_raw = Y_state[:train_end]
    Y_train_attack = Y_attack[:train_end]
    Y_train_stage = Y_stage[:train_end]
    
    X_val_raw = X_seq[train_end:val_end]
    Y_val_state_raw = Y_state[train_end:val_end]
    Y_val_attack = Y_attack[train_end:val_end]
    Y_val_stage = Y_stage[train_end:val_end]
    
    X_test_raw = X_seq[val_end:]
    Y_test_state_raw = Y_state[val_end:]
    Y_test_attack = Y_attack[val_end:]
    Y_test_stage = Y_stage[val_end:]
    
    # --- Fit Scaler ONLY on Training Sequences (reshaped to 2D for StandardScaler) ---
    N_tr, T_tr, D_tr = X_train_raw.shape
    X_train_2d = X_train_raw.reshape(-1, D_tr)
    
    normalizer = FeatureNormalizer()
    X_train_2d_scaled = normalizer.fit_transform(X_train_2d, feature_cols=feature_cols)
    X_train_seq = X_train_2d_scaled.reshape(N_tr, T_tr, D_tr)
    Y_train_state = normalizer.transform(Y_train_state_raw)
    
    # Transform Validation sequences using fitted scaler
    N_val, T_val, D_val = X_val_raw.shape
    X_val_2d_scaled = normalizer.transform(X_val_raw.reshape(-1, D_val))
    X_val_seq = X_val_2d_scaled.reshape(N_val, T_val, D_val)
    Y_val_state = normalizer.transform(Y_val_state_raw)
    
    # Transform Test sequences using fitted scaler
    N_te, T_te, D_te = X_test_raw.shape
    X_test_2d_scaled = normalizer.transform(X_test_raw.reshape(-1, D_te))
    X_test_seq = X_test_2d_scaled.reshape(N_te, T_te, D_te)
    Y_test_state = normalizer.transform(Y_test_state_raw)
    
    # Save Scaler
    normalizer.save(models_dir=models_dir, scaler_filename="scaler.pkl", cols_filename="feature_columns.pkl")
    
    # Save NumPy Arrays
    np.save(output_dir / "X_train_seq.npy", X_train_seq)
    np.save(output_dir / "Y_train_state.npy", Y_train_state)
    np.save(output_dir / "Y_train_attack.npy", Y_train_attack)
    np.save(output_dir / "Y_train_stage.npy", Y_train_stage)
    
    np.save(output_dir / "X_val_seq.npy", X_val_seq)
    np.save(output_dir / "Y_val_state.npy", Y_val_state)
    np.save(output_dir / "Y_val_attack.npy", Y_val_attack)
    np.save(output_dir / "Y_val_stage.npy", Y_val_stage)
    
    np.save(output_dir / "X_test_seq.npy", X_test_seq)
    np.save(output_dir / "Y_test_state.npy", Y_test_state)
    np.save(output_dir / "Y_test_attack.npy", Y_test_attack)
    np.save(output_dir / "Y_test_stage.npy", Y_test_stage)
    
    logger.info(f"Saved normalized sequence datasets to {output_dir}")
    logger.info(f"Splits -> Train: {len(X_train_seq)}, Val: {len(X_val_seq)}, Test: {len(X_test_seq)}")
    
    return {
        "X_train": X_train_seq, "Y_train_state": Y_train_state, "Y_train_attack": Y_train_attack, "Y_train_stage": Y_train_stage,
        "X_val": X_val_seq, "Y_val_state": Y_val_state, "Y_val_attack": Y_val_attack, "Y_val_stage": Y_val_stage,
        "X_test": X_test_seq, "Y_test_state": Y_test_state, "Y_test_attack": Y_test_attack, "Y_test_stage": Y_test_stage
    }

if __name__ == "__main__":
    create_temporal_sequences()
