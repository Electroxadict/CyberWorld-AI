"""
Unit tests for CyberWorld-AI Streamlit Dashboard UI components and Plotly chart generators.
"""

import sys
from pathlib import Path
import pytest
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.charts import (
    create_risk_gauge_chart,
    create_future_risk_chart,
    create_attack_prob_chart,
    create_shap_bar_chart,
    create_shap_group_chart,
    create_attention_chart
)
from dashboard.components import load_pipeline_cached

def test_app_import():
    """Verifies main app.py module can be imported without syntax/compilation errors."""
    import app
    assert hasattr(app, "main")

def test_plotly_chart_generators():
    """Verifies Plotly chart functions return valid Figure objects."""
    # 1. Gauge chart
    fig_g = create_risk_gauge_chart(45.5)
    assert isinstance(fig_g, go.Figure)
    
    # 2. Future risk chart
    fig_r = create_future_risk_chart([45.5, 46.0, 48.0, 50.0, 52.0])
    assert isinstance(fig_r, go.Figure)
    assert len(fig_r.data) > 0
    
    # 3. Attack probability chart
    fig_p = create_attack_prob_chart(0.64, [0.65, 0.66, 0.67, 0.68, 0.70])
    assert isinstance(fig_p, go.Figure)
    
    # 4. SHAP bar chart
    dummy_shap = [
        {"feature": "fut_slope_Src Port_std", "value": 0.0, "shap_value": 0.85, "direction": "INCREASES_RISK", "importance": 0.85},
        {"feature": "fut_diff_Port_Diversity", "value": -0.2, "shap_value": -0.24, "direction": "DECREASES_RISK", "importance": 0.24}
    ]
    fig_s = create_shap_bar_chart(dummy_shap, top_n=10)
    assert isinstance(fig_s, go.Figure)
    
    # 5. Attention chart
    timesteps = ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"]
    weights = [0.1] * 10
    fig_a = create_attention_chart(timesteps, weights, "t-3")
    assert isinstance(fig_a, go.Figure)

def test_pipeline_cached_loader():
    """Verifies cached pipeline loader returns initialized PCAPPredictivePipeline instance."""
    pipeline = load_pipeline_cached()
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "feature_extractor")
