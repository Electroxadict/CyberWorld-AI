"""
Smoke test suite for end-to-end CyberWorld-AI baseline training pipeline.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.check_dataset import check_dataset
from preprocessing.preprocess import preprocess_all
from preprocessing.time_window import aggregate_time_windows
from training.train_logistic import train_logistic_baseline
from training.train_random_forest import train_random_forest
from training.create_sequences import create_temporal_sequences

def test_full_baseline_pipeline():
    # 1. Dataset check
    check_dataset()
    
    # 2. Preprocess
    proc_path = preprocess_all()
    assert Path(proc_path).exists()
    
    # 3. Time Windows
    df_win = aggregate_time_windows()
    assert len(df_win) > 0
    
    # 4. Logistic Baseline
    clf_log, metrics_log = train_logistic_baseline()
    assert metrics_log["Accuracy"] >= 0.0
    
    # 5. Random Forest Baseline
    clf_rf, metrics_rf, importances = train_random_forest()
    assert len(importances) > 0
    
    # 6. Temporal Sequence Creation
    seq_dict = create_temporal_sequences()
    assert "X_train" in seq_dict
    assert seq_dict["X_train"].ndim == 3 # (N, seq_len, num_features)
