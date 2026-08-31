"""
Unit and Integration tests for CyberWorld-AI PCAP Feature Extraction and Ingestion Pipeline.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.feature_extractor import PCAPFeatureExtractor
from inference.pcap_pipeline import PCAPPredictivePipeline
from scripts.generate_sample_pcap import generate_sample_pcap

def test_pcap_extractor_validation(tmp_path):
    extractor = PCAPFeatureExtractor()
    
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "non_existent.pcap")
        
    # 2. Unsupported extension
    invalid_txt = tmp_path / "test.txt"
    invalid_txt.write_text("dummy")
    with pytest.raises(ValueError):
        extractor.extract(invalid_txt)

def test_pcap_sample_extraction(tmp_path):
    pcap_path = tmp_path / "test_sample.pcap"
    generate_sample_pcap(pcap_path)
    
    extractor = PCAPFeatureExtractor()
    df_flows = extractor.extract(pcap_path)
    
    assert len(df_flows) > 0
    assert "Timestamp" in df_flows.columns
    assert "Src Port" in df_flows.columns
    assert "Dst Port" in df_flows.columns
    assert "Flow Duration" in df_flows.columns

def test_insufficient_history_handling(tmp_path):
    pcap_path = tmp_path / "tiny.pcap"
    generate_sample_pcap(pcap_path)
    
    pipeline = PCAPPredictivePipeline()
    df_flows = pipeline.extract_features(pcap_path, max_packets=5)
    df_windows = pipeline.create_temporal_windows(df_flows)
    scaled_matrix = pipeline.validate_and_scale_features(df_windows)
    
    # Tiny packet sample will have < 10 windows
    if len(scaled_matrix) < 10:
        with pytest.raises(ValueError, match="Insufficient temporal history"):
            pipeline.create_sequence(scaled_matrix)

def test_end_to_end_pcap_pipeline(tmp_path):
    pcap_path = tmp_path / "test_full.pcap"
    generate_sample_pcap(pcap_path)
    
    pipeline = PCAPPredictivePipeline()
    res = pipeline.predict(pcap_path)
    
    assert res["source"] == "PCAP"
    assert res["flow_count"] > 0
    assert res["window_count"] >= 10
    assert "current_attack_probability" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert len(res["future_risk"]) == 5
    assert len(res["future_attack_probability"]) == 5
    assert "predicted_stage" in res
    assert "warning_message" in res
    assert len(res["top_shap_features"]) > 0
    assert len(res["attention_weights"]) == 10
