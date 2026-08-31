"""
Unit tests for XGBoost Risk and Attack Stage models and predictor interface.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.xgboost_predictor import XGBoostPredictor
from training.train_xgboost import build_xgb_feature_names

def test_xgb_feature_name_construction():
    base_cols = ["feat_A", "feat_B"]
    feat_names = build_xgb_feature_names(base_cols)
    
    # 2 base + (2 * 6 engineered) + 6 summary = 2 + 12 + 6 = 20 features
    assert len(feat_names) == 20
    assert "curr_feat_A" in feat_names
    assert "fut_mean_feat_A" in feat_names
    assert "wm_current_attack_prob" in feat_names

def test_xgboost_predictor_inference():
    models_dir = PROJECT_ROOT / "models"
    if not (models_dir / "xgb_risk_model.pkl").exists():
        pytest.skip("xgb_risk_model.pkl not found. Run train_xgboost.py first.")
        
    predictor = XGBoostPredictor(models_dir=models_dir)
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    
    res = predictor.predict(dummy_seq)
    
    assert "attack_probability" in res
    assert 0.0 <= res["attack_probability"] <= 1.0
    assert "predicted_stage" in res
    assert 0 <= res["predicted_stage"] <= 5
    assert len(res["stage_probabilities"]) == 6
    assert pytest.approx(sum(res["stage_probabilities"]), 0.01) == 1.0
    assert "stage_name" in res
    assert res["model"] == "XGBoost"
