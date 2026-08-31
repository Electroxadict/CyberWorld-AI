"""
Streamlit UI Components and Helpers for CyberWorld-AI Control Room Dashboard.
Provides dark SOC CSS injection, cached model pipeline initialization, KPI cards,
and visual status panels.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import xgboost
import shap
import scapy
import plotly

from inference.pcap_pipeline import PCAPPredictivePipeline

@st.cache_resource(show_spinner="Loading CyberWorld-AI ML Models & Explainers...")
def load_pipeline_cached(models_dir=None):
    """Loads and caches PCAPPredictivePipeline to prevent reloading models on every interaction."""
    return PCAPPredictivePipeline(models_dir=models_dir)

def inject_dark_soc_css():
    """Injects dark cybersecurity SOC control-room styling into Streamlit app."""
    css = """
    <style>
    /* Dark SOC Theme Colors */
    .stApp {
        background-color: #0E1117;
        color: #F9FAFB;
    }
    
    /* Header Card */
    .soc-header {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    .soc-title {
        color: #60A5FA;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .soc-subtitle {
        color: #9CA3AF;
        font-size: 14px;
        margin-top: 4px;
    }
    
    /* System Status Pills */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 8px;
    }
    
    .pill-green { background-color: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .pill-blue { background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }
    .pill-yellow { background-color: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .pill-red { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    
    /* Main Status Banner */
    .status-banner-normal {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-left: 6px solid #10B981;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    .status-banner-warning {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        border-left: 6px solid #F59E0B;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    .status-banner-critical {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        border-left: 6px solid #EF4444;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    /* Metric Card Custom Styling */
    div[data-testid="stMetric"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_header():
    """Renders dashboard top header."""
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-title">CyberWorld-AI — Predictive Cyber Defence Platform</h1>
        <div class="soc-subtitle">Temporal World Model + XGBoost + Explainable AI (SHAP & Temporal Attention)</div>
    </div>
    """, unsafe_allow_html=True)

def render_status_bar(device_name="CPU"):
    """Renders system status indicators."""
    st.markdown(f"""
    <div style="margin-bottom: 16px;">
        <span class="status-pill pill-green">● SYSTEM ONLINE</span>
        <span class="status-pill pill-blue">PCAP Engine: READY</span>
        <span class="status-pill pill-blue">World Model: READY</span>
        <span class="status-pill pill-blue">XGBoost: READY</span>
        <span class="status-pill pill-blue">Explainability: READY</span>
        <span class="status-pill pill-yellow">Device: {device_name}</span>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(res: dict):
    """Renders top KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Current Attack Probability",
            value=f"{res['current_attack_probability'] * 100:.1f}%"
        )
        
    with col2:
        st.metric(
            label="Risk Score",
            value=f"{res['risk_score']:.1f} / 100"
        )
        
    with col3:
        st.metric(
            label="Risk Level",
            value=res["risk_level"]
        )
        
    with col4:
        st.metric(
            label="Predicted Stage",
            value=res["stage_name"]
        )
        
    with col5:
        t_high = res["time_to_high_risk"]
        st.metric(
            label="Time to High Risk",
            value=f"~{t_high}s" if t_high is not None else "N/A (Low)"
        )

def render_main_status_panel(res: dict):
    """Renders prominent security status alert panel."""
    level = res["risk_level"]
    msg = res["warning_message"]
    
    if level in ["HIGH", "CRITICAL"] or res["warning_triggered"]:
        banner_class = "status-banner-critical"
        status_text = "CRITICAL / WARNING — ELEVATED ATTACK PROGRESSION DETECTED"
    elif level == "MODERATE":
        banner_class = "status-banner-warning"
        status_text = "MODERATE RISK — ANOMALOUS ACTIVITY MONITORING"
    else:
        banner_class = "status-banner-normal"
        status_text = "NORMAL OPERATIONAL BASELINE — SYSTEM SECURE"
        
    st.markdown(f"""
    <div class="{banner_class}">
        <h3 style="margin: 0 0 8px 0; font-size: 18px; font-weight: 700;">{status_text}</h3>
        <p style="margin: 0; font-size: 14px;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline_flow():
    """Renders visual architecture pipeline flow diagram."""
    st.markdown("""
    ```
    PCAP → Scapy Extractor → 5-sec Windows → 69 Features → LSTM World Model → 5-step Rollout → XGBoost → Risk Engine → SHAP & Attention
    [READY]    [READY]         [READY]          [READY]         [READY]            [READY]        [READY]     [READY]          [READY]
    ```
    """)

def render_model_info_expander(config: dict):
    """Renders expandable system & model information section."""
    with st.expander("System & Model Architecture Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model & Feature Specifications:**")
            st.markdown(f"- **Base Network State Features**: 69 features")
            st.markdown(f"- **XGBoost Engineered Features**: 489 features")
            st.markdown(f"- **Temporal Sequence Window**: 10 states (50 seconds total history)")
            st.markdown(f"- **Time Window Aggregation**: 5 seconds")
            st.markdown(f"- **Prediction Rollout Horizon**: 5 steps (25 seconds future)")
            st.markdown(f"- **LSTM World Model**: 2-layer LSTM (`hidden_size=128`) + Temporal Attention")
            st.markdown(f"- **XGBoost Classifier**: Risk Model & 6-class MITRE Stage Model")
            st.markdown(f"- **Explainability Engines**: TreeSHAP + Temporal Attention")
        with col2:
            st.markdown("**Software Dependencies & Versions:**")
            st.markdown(f"- **Streamlit**: v{st.__version__}")
            st.markdown(f"- **Plotly**: v{plotly.__version__}")
            st.markdown(f"- **PyTorch**: v{torch.__version__}")
            st.markdown(f"- **XGBoost**: v{xgboost.__version__}")
            st.markdown(f"- **SHAP**: v{shap.__version__}")
            st.markdown(f"- **Scapy**: v{scapy.__version__}")
            st.markdown(f"- **Execution Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
