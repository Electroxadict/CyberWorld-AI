"""
Unit tests for CyberWorld-AI feature extraction and time window aggregation.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.check_dataset import generate_synthetic_dataset
from preprocessing.preprocess import clean_dataframe
from preprocessing.time_window import aggregate_time_windows

def test_synthetic_dataset_generation(tmp_path):
    output_path = tmp_path / "test_raw.csv"
    gen_path = generate_synthetic_dataset(output_path)
    assert gen_path.exists()
    df = pd.read_csv(gen_path)
    assert len(df) > 0
    assert "Label" in df.columns
    assert "Timestamp" in df.columns

def test_preprocessing_and_time_window(tmp_path):
    raw_path = tmp_path / "test_raw.csv"
    generate_synthetic_dataset(raw_path)
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_dataframe(df_raw)
    
    assert "Attack" in df_clean.columns
    assert "Attack_Stage" in df_clean.columns
    assert df_clean["Attack"].isin([0, 1]).all()
    
    # Save clean dataset to tmp
    clean_csv_path = tmp_path / "test_processed.csv"
    df_clean.to_csv(clean_csv_path, index=False)
    
    # Run time windowing
    df_windows = aggregate_time_windows(processed_csv_path=clean_csv_path)
    assert len(df_windows) > 0
    assert "Flow_Count" in df_windows.columns
    assert "Port_Diversity" in df_windows.columns
    assert "Traffic_Burst_Score" in df_windows.columns
    assert "Attack" in df_windows.columns
