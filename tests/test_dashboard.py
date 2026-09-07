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

def test_render_live_connection_table():
    """Verifies live connection table produces valid HTML without 4-space markdown indentations."""
    from unittest.mock import patch
    import pandas as pd
    from dashboard.components import render_live_connection_table

    df_sample = pd.DataFrame([
        {
            "Source IP": "192.168.1.105",
            "Destination IP": "10.0.0.1",
            "Protocol": "TCP",
            "Source Port": 54321,
            "Destination Port": 443,
            "Packets": 150,
            "Bytes": 12500,
            "Duration": "2.4s",
            "Status": "Normal"
        },
        {
            "Source IP": "192.168.1.200",
            "Destination IP": "10.0.0.5",
            "Protocol": "UDP",
            "Source Port": 38570,
            "Destination Port": 53,
            "Packets": 5000,
            "Bytes": 450000,
            "Duration": "15.1s",
            "Status": "Critical"
        }
    ])

    with patch("streamlit.markdown") as mock_md:
        render_live_connection_table(df_sample)
        assert mock_md.called
        call_args = mock_md.call_args
        rendered_html = call_args[0][0]
        kwargs = call_args[1]

        # Verify unsafe_allow_html is True
        assert kwargs.get("unsafe_allow_html") is True

        # Verify essential HTML components
        assert '<div class="live-table-container">' in rendered_html
        assert '<table' in rendered_html
        assert '<thead' in rendered_html
        assert '<tbody>' in rendered_html
        assert '192.168.1.105' in rendered_html
        assert 'status-normal' in rendered_html
        assert 'status-critical' in rendered_html
        assert '12,500' in rendered_html

        # Verify that no lines in rendered_html start with 4 or more spaces (which triggers CommonMark code blocks)
        for line in rendered_html.split("\n"):
            assert not line.startswith("    "), f"Indented line found which triggers markdown code block: {line}"

