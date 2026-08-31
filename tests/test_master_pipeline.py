"""
Unit and Integration tests for CyberWorld-AI Master Pipeline and CLI Verification Tools.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_environment import check_env
from scripts.validate_models import validate_all_models
from scripts.generate_sample_pcap import generate_sample_pcap
from inference.pcap_pipeline import PCAPPredictivePipeline

def test_environment_checker():
    """Verifies check_environment.py executes without error."""
    assert check_env() is True

def test_model_validator():
    """Verifies validate_models.py verifies model artifact integrity."""
    assert validate_all_models() is True

def test_master_pipeline_execution(tmp_path):
    """Verifies end-to-end master pipeline prediction and dictionary keys."""
    sample_pcap = tmp_path / "master_test.pcap"
    generate_sample_pcap(sample_pcap)
    
    pipeline = PCAPPredictivePipeline()
    res = pipeline.predict(sample_pcap)
    
    assert res["source"] == "PCAP"
    assert "current_attack_probability" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert "future_risk" in res
    assert "top_shap_features" in res
    assert "attention_weights" in res
