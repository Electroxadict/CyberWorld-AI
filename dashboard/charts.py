"""
Plotly Chart Generators for CyberWorld-AI Streamlit Dashboard.
Creates dark SOC control-room interactive charts for risk gauges, forecasts, SHAP feature contributions, and temporal attention.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Dark SOC theme palette
BG_COLOR = "#111827"
CARD_BG = "#1F2937"
TEXT_COLOR = "#F9FAFB"
GRID_COLOR = "#374151"

def create_risk_gauge_chart(risk_score: float) -> go.Figure:
    """Generates 0-100 Plotly Gauge chart for current risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Risk Score (0-100)", 'font': {'size': 18, 'color': TEXT_COLOR}},
        number={'suffix': " / 100", 'font': {'size': 24, 'color': TEXT_COLOR}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_COLOR},
            'bar': {'color': "#3B82F6" if risk_score < 40 else ("#F59E0B" if risk_score < 70 else "#EF4444")},
            'bgcolor': CARD_BG,
            'borderwidth': 1,
            'bordercolor': GRID_COLOR,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},   # Low (Green)
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},  # Moderate (Yellow)
                {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.2)'},  # High (Orange)
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}   # Critical (Red)
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={'color': TEXT_COLOR},
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig

def create_future_risk_chart(future_risks: list) -> go.Figure:
    """Generates line chart for 5-step recursive World Model future risk forecast."""
    steps = ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"]
    values = [future_risks[0]] + list(future_risks) if len(future_risks) == 5 else list(future_risks)
    if len(values) > 6:
        values = values[:6]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode="lines+markers",
        name="Predicted Risk",
        line=dict(color="#F59E0B", width=3),
        marker=dict(size=8, color="#F59E0B")
    ))
    fig.update_layout(
        title="5-Step World Model Future Risk Trajectory",
        xaxis_title="Simulation Horizon",
        yaxis_title="Predicted Risk Score (0-100)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig

def create_attack_prob_chart(current_prob: float, future_probs: list) -> go.Figure:
    """Generates bar/line chart for predicted attack probability progression."""
    steps = ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"]
    probs = [current_prob * 100.0] + [p * 100.0 for p in future_probs]
    if len(probs) > 6:
        probs = probs[:6]
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps,
        y=probs,
        name="Attack Probability (%)",
        marker_color=["#10B981" if p < 40 else ("#F59E0B" if p < 70 else "#EF4444") for p in probs]
    ))
    fig.update_layout(
        title="Predicted Attack Probability Progression (%)",
        xaxis_title="Simulation Horizon",
        yaxis_title="Attack Probability (%)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig

def create_shap_bar_chart(top_shap_features: list, top_n: int = 10) -> go.Figure:
    """Generates horizontal bar chart of local SHAP feature contributions."""
    display_feats = top_shap_features[:top_n][::-1]
    
    names = [f["feature"] for f in display_feats]
    s_vals = [f["shap_value"] for f in display_feats]
    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in s_vals]
    
    fig = go.Figure(go.Bar(
        x=s_vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in s_vals],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Top {len(names)} Local Feature Contributions (SHAP Explanation)",
        xaxis_title="SHAP Contribution (+ Increases Risk / - Decreases Risk)",
        yaxis_title="Feature Name",
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=150, r=20, t=50, b=40),
        height=400
    )
    return fig

def create_shap_group_chart(group_importance: dict) -> go.Figure:
    """Generates bar chart of SHAP macro feature group share percentages."""
    groups = list(group_importance.keys())
    shares = [group_importance[g]["percentage_share"] for g in groups]
    
    fig = go.Figure(go.Bar(
        x=groups,
        y=shares,
        marker_color="#8B5CF6",
        text=[f"{s:.1f}%" for s in shares],
        textposition="auto"
    ))
    fig.update_layout(
        title="SHAP Macro Feature Category Importance Share (%)",
        xaxis_title="Feature Category",
        yaxis_title="Importance Share (%)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def create_attention_chart(timesteps: list, weights: list, max_label: str) -> go.Figure:
    """Generates bar chart of PyTorch World Model temporal window attention weights."""
    colors = ["#EF4444" if t == max_label else "#3B82F6" for t in timesteps]
    pct_weights = [w * 100.0 for w in weights]
    
    fig = go.Figure(go.Bar(
        x=timesteps,
        y=pct_weights,
        marker_color=colors,
        text=[f"{pw:.1f}%" for pw in pct_weights],
        textposition="auto"
    ))
    fig.update_layout(
        title="PyTorch LSTM World Model Temporal Window Attention (t-9 ... t)",
        xaxis_title="Historical Network State Windows",
        yaxis_title="Attention Weight (%)",
        yaxis=dict(range=[0, max(pct_weights) * 1.3], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig
