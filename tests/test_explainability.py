"""
Unit tests for CyberWorld-AI SHAP Explainer and Temporal Attention Visualizer.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.shap_explainer import SHAPExplainer
from explainability.attention_visualizer import TemporalAttentionVisualizer

def test_shap_explainer():
    models_dir = PROJECT_ROOT / "models"
    if not (models_dir / "xgb_risk_model.pkl").exists():
        pytest.skip("xgb_risk_model.pkl not found. Run train_xgboost.py first.")
        
    explainer = SHAPExplainer(models_dir=models_dir)
    dummy_x = np.random.randn(20, 489).astype(np.float32)
    
    # 1. Global SHAP
    global_imp = explainer.compute_global_shap(dummy_x, sample_size=10)
    assert len(global_imp) == 489
    assert "feature" in global_imp[0]
    assert "mean_abs_shap" in global_imp[0]
    
    # 2. Local SHAP
    local_exp = explainer.explain_prediction(dummy_x[0])
    assert len(local_exp) == 489
    assert "direction" in local_exp[0]
    assert local_exp[0]["direction"] in ["INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"]
    
    # 3. Feature Group Aggregation
    group_imp = explainer.group_shap_features(dummy_x)
    assert len(group_imp) == 8
    assert "CURRENT" in group_imp
    assert "WORLD MODEL THREAT FEATURES" in group_imp
    
    # 4. Text Explanation
    text_exp = explainer.generate_explanation(dummy_x[0])
    assert "Prediction Threat Evaluation" in text_exp

def test_temporal_attention_visualizer():
    models_dir = PROJECT_ROOT / "models"
    if not (models_dir / "world_model.pt").exists():
        pytest.skip("world_model.pt not found. Run train_world_model.py first.")
        
    attn_vis = TemporalAttentionVisualizer()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    
    # 1. Attention Weights
    attn_res = attn_vis.get_temporal_attention(dummy_seq)
    assert len(attn_res["timesteps"]) == 10
    assert len(attn_res["attention_weights"]) == 10
    assert pytest.approx(sum(attn_res["attention_weights"]), 0.05) == 1.0
    
    # 2. Most Influential Timestep
    max_label, max_idx, max_weight = attn_vis.get_most_influential_timestep(dummy_seq)
    assert max_label in attn_res["timesteps"]
    assert 0 <= max_idx < 10
    assert max_weight > 0.0
    
    # 3. Temporal Explanation Text
    temp_exp = attn_vis.generate_temporal_explanation(dummy_seq)
    assert "World Model placed the highest temporal attention" in temp_exp
