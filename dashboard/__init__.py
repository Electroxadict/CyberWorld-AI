"""
Dashboard package for CyberWorld-AI.
Contains reusable UI components and Plotly charts for the Streamlit control room interface.
"""

from dashboard.components import load_pipeline_cached, render_header, render_status_bar, render_kpi_cards
from dashboard.charts import (
    create_risk_gauge_chart,
    create_future_risk_chart,
    create_attack_prob_chart,
    create_shap_bar_chart,
    create_shap_group_chart,
    create_attention_chart
)

__all__ = [
    "load_pipeline_cached",
    "render_header",
    "render_status_bar",
    "render_kpi_cards",
    "create_risk_gauge_chart",
    "create_future_risk_chart",
    "create_attack_prob_chart",
    "create_shap_bar_chart",
    "create_shap_group_chart",
    "create_attention_chart"
]
