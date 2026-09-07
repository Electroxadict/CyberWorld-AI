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
import pandas as pd
import numpy as np
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
    
    /* Metric Card Custom Styling & Animations */
    @keyframes kpi-pulse {
        0% { border-color: #374151; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        50% { border-color: #3B82F6; box-shadow: 0 0 10px rgba(59,130,246,0.3); }
        100% { border-color: #374151; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    }
    
    div[data-testid="stMetric"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #60A5FA;
    }

    /* Live Status Indicators */
    .live-badge-active {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        font-weight: 700;
        font-size: 13px;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
    }

    .live-badge-stopped {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        font-weight: 700;
        font-size: 13px;
    }

    /* Table status tag styles */
    .status-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .status-normal { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .status-suspicious { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .status-highrisk { background: rgba(249, 115, 22, 0.2); color: #F97316; border: 1px solid #F97316; }
    .status-critical { background: rgba(239, 68, 68, 0.25); color: #EF4444; border: 1px solid #EF4444; }

    /* Scrollable Live Table Container */
    .live-table-container {
        max-height: 380px;
        overflow-y: auto;
        border: 1px solid #374151;
        border-radius: 8px;
        background-color: #111827;
    }

    /* Responsive containers */
    @media (max-width: 1400px) {
        div[data-testid="stMetric"] {
            padding: 8px 10px;
        }
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
        if "time_to_high_risk_display" in res:
            t_display = res["time_to_high_risk_display"]
        else:
            t_high = res.get("time_to_high_risk")
            t_display = f"~{t_high}s" if t_high is not None else "N/A (Safe)"
            
        st.metric(
            label="Time to High Risk",
            value=t_display
        )

def render_main_status_panel(res: dict):
    """Renders prominent security status alert panel."""
    level = res["risk_level"]
    msg = res["warning_message"]
    sec_status = res.get("security_status", level)
    
    if sec_status in ["NETWORK COMPROMISED", "DANGER"] or level in ["HIGH", "CRITICAL"] or res.get("warning_triggered", False):
        banner_class = "status-banner-critical"
        status_text = f"SECURITY STATUS: {sec_status} — {res.get('warning_priority', 'ELEVATED ATTACK PROGRESSION DETECTED')}"
    elif sec_status in ["MILD ATTACK", "RISK"] or level == "MODERATE":
        banner_class = "status-banner-warning"
        status_text = f"SECURITY STATUS: {sec_status} — ANOMALOUS ACTIVITY MONITORING"
    else:
        banner_class = "status-banner-normal"
        status_text = f"SECURITY STATUS: {sec_status} — SYSTEM SECURE"
        
    st.markdown(f"""
    <div class="{banner_class}">
        <h3 style="margin: 0 0 8px 0; font-size: 18px; font-weight: 700;">{status_text}</h3>
        <p style="margin: 0; font-size: 14px;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

def render_mode_indicator(res: dict):
    """Renders prominent top badge indicating DEMO MODE vs REAL PCAP INFERENCE."""
    is_demo = res.get("is_demo_mode", False)
    if is_demo:
        mode_title = "🧪 DEMO MODE — SCENARIO SIMULATION"
        mode_subtitle = "Displaying progressive scenario simulation values for SIH demonstration. Feature drivers & attention weights derived from live model execution."
        badge_style = "background-color: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #F59E0B;"
    else:
        mode_title = "⚡ REAL PCAP INFERENCE"
        mode_subtitle = "Displaying live predictions, risk scores, and stage classifications computed directly by trained PyTorch & XGBoost ML models."
        badge_style = "background-color: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981;"
        
    full_html = f'<div style="{badge_style} border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;"><div style="font-weight: 700; font-size: 15px;">{mode_title}</div><div style="font-size: 12px; opacity: 0.9;">{mode_subtitle}</div></div>'
    st.markdown(full_html, unsafe_allow_html=True)

def render_progression_stepper(res: dict):
    """Renders visual scenario progression stepper for SIH presentation."""
    current_step = res.get("progression_step", 1)
    
    steps = [
        ("1. SAFE", "LOW"),
        ("2. MILD ATTACK", "MODERATE"),
        ("3. RISK", "HIGH"),
        ("4. DANGER", "CRITICAL"),
        ("5. NETWORK COMPROMISED", "CRITICAL")
    ]
    
    colors = {
        1: ("#10B981", "rgba(16, 185, 129, 0.2)"), # Green
        2: ("#F59E0B", "rgba(245, 158, 11, 0.2)"), # Amber
        3: ("#F97316", "rgba(249, 115, 22, 0.2)"), # Orange
        4: ("#EF4444", "rgba(239, 68, 68, 0.2)"), # Red
        5: ("#DC2626", "rgba(220, 38, 38, 0.3)")  # Deep Red
    }
    
    items_html = []
    for idx, (label, level_tag) in enumerate(steps, start=1):
        main_color, bg_color = colors[idx]
        
        if idx == current_step:
            # ACTIVE STATE
            item_str = f'<div style="flex: 1; min-width: 130px; background: {bg_color}; border: 2px solid {main_color}; color: {main_color}; border-radius: 8px; padding: 10px 6px; text-align: center; font-weight: 700; box-shadow: 0 0 12px {main_color}50;"><div style="font-size: 13px; letter-spacing: 0.5px;">{label}</div><div style="font-size: 11px; font-weight: 800; margin-top: 4px; color: {main_color};">● ACTIVE ✓</div></div>'
        elif idx < current_step:
            # COMPLETED STATE
            item_str = f'<div style="flex: 1; min-width: 130px; background: rgba(31, 41, 55, 0.6); border: 1px solid #4B5563; color: #9CA3AF; border-radius: 8px; padding: 10px 6px; text-align: center; font-weight: 600;"><div style="font-size: 13px; text-decoration: line-through; opacity: 0.8;">{label}</div><div style="font-size: 10px; margin-top: 4px; color: #10B981;">✓ PASSED</div></div>'
        else:
            # INACTIVE / FUTURE STATE
            item_str = f'<div style="flex: 1; min-width: 130px; background: #111827; border: 1px dashed #374151; color: #4B5563; border-radius: 8px; padding: 10px 6px; text-align: center;"><div style="font-size: 13px;">{label}</div><div style="font-size: 10px; margin-top: 4px; color: #6B7280;">○ PENDING</div></div>'
            
        items_html.append(item_str)
        
        if idx < len(steps):
            sep_color = "#10B981" if idx < current_step else "#374151"
            items_html.append(f'<div style="color: {sep_color}; font-size: 16px; font-weight: 700; padding: 0 2px;">➔</div>')
            
    full_html = '<div style="display: flex; gap: 6px; margin-bottom: 24px; align-items: center; justify-content: space-between; overflow-x: auto; padding: 4px 0;">' + "".join(items_html) + '</div>'
    st.markdown(full_html, unsafe_allow_html=True)

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

def render_inference_debug_info(res: dict, target_pcap_path: Path):
    """Renders expandable Inference Debug & Lineage Information section (Part 5 requirement)."""
    with st.expander("🔍 Inference Debug & Lineage Information", expanded=True):
        col1, col2 = st.columns(2)
        
        import hashlib
        md5_hash = "N/A"
        file_size_kb = 0.0
        if target_pcap_path and target_pcap_path.exists():
            try:
                md5_hash = hashlib.md5(target_pcap_path.read_bytes()).hexdigest()
                file_size_kb = target_pcap_path.stat().st_size / 1024.0
            except Exception:
                pass
                
        with col1:
            st.markdown("**PCAP Telemetry & Lineage Details:**")
            st.markdown(f"- **PCAP Filename**: `{target_pcap_path.name if target_pcap_path else res.get('pcap_file', 'N/A')}`")
            st.markdown(f"- **File Size**: `{file_size_kb:.1f} KB`")
            st.markdown(f"- **MD5 Hash**: `{md5_hash}`")
            st.markdown(f"- **Total Flow Count**: `{res.get('flow_count', 0)}`")
            st.markdown(f"- **Temporal 5s Windows**: `{res.get('window_count', 0)}`")
            st.markdown(f"- **Base Feature Shape**: `(10, 69)` (50s Context)")
            st.markdown(f"- **XGBoost Feature Vector**: `(1, 489)` Features")

        with col2:
            st.markdown("**Runtime Inference Environment:**")
            st.markdown(f"- **Inference Timestamp**: `{res.get('inference_timestamp', 'N/A')}`")
            st.markdown(f"- **Pipeline Execution Time**: `{res.get('execution_time_seconds', 0.0):.3f} seconds`")
            st.markdown(f"- **World Model Architecture**: PyTorch 2-Layer LSTM + Attention Core")
            st.markdown(f"- **Risk & Stage Predictors**: XGBoost Risk Model & XGBoost 6-Class Stage Model")
            st.markdown(f"- **Explainability Engines**: TreeSHAP (Local & Group) + PyTorch Attention")
            st.markdown(f"- **Execution Device**: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU (Fallback)'}")

# --- RESPONSIVE SOC DASHBOARD COMPONENTS ---

def render_threat_level_progression(risk_score: float):
    """
    Renders horizontal threat level progression highlighting active level.
    Risk mapping:
    0–20: SAFE
    21–40: MILD ATTACK
    41–60: RISK
    61–80: DANGER
    81–100: NETWORK COMPROMISED
    """
    if risk_score <= 20:
        active_idx = 1
        curr_label = "SAFE"
    elif risk_score <= 40:
        active_idx = 2
        curr_label = "MILD ATTACK"
    elif risk_score <= 60:
        active_idx = 3
        curr_label = "RISK"
    elif risk_score <= 80:
        active_idx = 4
        curr_label = "DANGER"
    else:
        active_idx = 5
        curr_label = "NETWORK COMPROMISED"

    stages = [
        (1, "SAFE", "0-20", "#10B981", "rgba(16, 185, 129, 0.2)"),
        (2, "MILD ATTACK", "21-40", "#F59E0B", "rgba(245, 158, 11, 0.2)"),
        (3, "RISK", "41-60", "#F97316", "rgba(249, 115, 22, 0.2)"),
        (4, "DANGER", "61-80", "#EF4444", "rgba(239, 68, 68, 0.25)"),
        (5, "NETWORK COMPROMISED", "81-100", "#DC2626", "rgba(220, 38, 38, 0.3)")
    ]

    items_html = []
    for idx, name, rng, border_col, bg_col in stages:
        if idx == active_idx:
            # ACTIVE HIGHLIGHT
            item = f'<div style="flex: 1; min-width: 140px; background: {bg_col}; border: 2px solid {border_col}; border-radius: 8px; padding: 8px 6px; text-align: center; font-weight: 700; color: {border_col}; box-shadow: 0 0 14px {border_col}60;"><div style="font-size: 13px; letter-spacing: 0.5px;">{name}</div><div style="font-size: 11px; opacity: 0.9; margin-top: 2px;">{rng} ● ACTIVE</div></div>'
        elif idx < active_idx:
            # PREVIOUS / PASSED LEVEL
            item = f'<div style="flex: 1; min-width: 140px; background: rgba(31, 41, 55, 0.5); border: 1px solid #4B5563; border-radius: 8px; padding: 8px 6px; text-align: center; color: #9CA3AF;"><div style="font-size: 12px;">{name}</div><div style="font-size: 10px; color: #10B981; margin-top: 2px;">✓ BELOW</div></div>'
        else:
            # FUTURE LEVEL
            item = f'<div style="flex: 1; min-width: 140px; background: #111827; border: 1px dashed #374151; border-radius: 8px; padding: 8px 6px; text-align: center; color: #6B7280;"><div style="font-size: 12px;">{name}</div><div style="font-size: 10px; margin-top: 2px;">{rng}</div></div>'
            
        items_html.append(item)

    full_html = '<div style="display: flex; gap: 8px; margin-bottom: 20px; align-items: center; justify-content: space-between; overflow-x: auto; padding: 2px 0;">' + "".join(items_html) + '</div>'
    st.markdown(full_html, unsafe_allow_html=True)

def render_responsive_kpi_row(metrics: dict, res: dict):
    """
    Renders top row of 8 responsive live KPI metric cards.
    Displays:
    1. Active Connections
    2. Packets/sec
    3. Bandwidth (Mbps)
    4. Total Flows
    5. Risk Score
    6. Attack Probability
    7. Current MITRE Stage
    8. Threat Level
    """
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    # Extract live or pipeline metrics
    active_conns = metrics.get("active_connections", res.get("flow_count", 0))
    pps = metrics.get("packets_per_sec", 0.0)
    mbps = metrics.get("bandwidth_mbps", 0.0)
    tot_flows = metrics.get("total_flows", res.get("flow_count", 0))
    
    risk_score = res.get("risk_score", 12.0)
    attack_prob = res.get("current_attack_probability", 0.05)
    mitre_stage = res.get("stage_name", "Normal")
    
    # Calculate Threat Level
    if risk_score <= 20:
        threat_level = "SAFE"
    elif risk_score <= 40:
        threat_level = "MILD ATTACK"
    elif risk_score <= 60:
        threat_level = "RISK"
    elif risk_score <= 80:
        threat_level = "DANGER"
    else:
        threat_level = "COMPROMISED"

    with col1:
        st.metric(label="Active Connections", value=f"{active_conns:,}")
    with col2:
        st.metric(label="Packets / Sec", value=f"{pps:.1f}")
    with col3:
        st.metric(label="Bandwidth (Mbps)", value=f"{mbps:.3f}")
    with col4:
        st.metric(label="Total Flows", value=f"{tot_flows:,}")
        
    with col5:
        st.metric(label="Risk Score (0-100)", value=f"{risk_score:.1f} / 100")
    with col6:
        st.metric(label="Attack Probability", value=f"{attack_prob * 100.0:.1f}%")
    with col7:
        st.metric(label="Current MITRE Stage", value=mitre_stage)
    with col8:
        st.metric(label="Threat Level", value=threat_level)

def render_mitre_panel(predicted_stage: int, stage_probabilities: list = None):
    """
    Renders MITRE ATT&CK Matrix panel highlighting detected stage with probability beside each.
    Stages:
    - Normal (0)
    - Reconnaissance (1)
    - Initial Access (2)
    - Lateral Movement (3)
    - Command & Control (4)
    - Exfiltration (5)
    """
    stages = [
        (0, "Normal", "Benign baseline activity"),
        (1, "Reconnaissance", "Port scanning & discovery"),
        (2, "Initial Access", "Brute-force / entry attempts"),
        (3, "Lateral Movement", "Internal pivoting / DoS"),
        (4, "Command & Control", "C2 beaconing & botnet"),
        (5, "Exfiltration", "Data theft / anomalous upload")
    ]
    
    if not stage_probabilities or len(stage_probabilities) < 6:
        stage_probabilities = [0.0] * 6
        if 0 <= predicted_stage < 6:
            stage_probabilities[predicted_stage] = 0.85
            
    items_html = []
    for s_idx, s_name, s_desc in stages:
        p_val = stage_probabilities[s_idx] * 100.0 if s_idx < len(stage_probabilities) else 0.0
        
        if s_idx == predicted_stage:
            card_html = f"""
            <div style="background: rgba(239, 68, 68, 0.2); border: 2px solid #EF4444; border-radius: 8px; padding: 12px; margin-bottom: 8px; box-shadow: 0 0 10px rgba(239,68,68,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 700; color: #EF4444; font-size: 15px;">Stage {s_idx}: {s_name}</div>
                    <div style="background: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 12px;">{p_val:.1f}% ● DETECTED</div>
                </div>
                <div style="font-size: 12px; color: #D1D5DB; margin-top: 4px;">{s_desc}</div>
            </div>
            """
        else:
            card_html = f"""
            <div style="background: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #9CA3AF; font-size: 14px;">Stage {s_idx}: {s_name}</div>
                    <div style="color: #60A5FA; font-weight: 600; font-size: 12px;">{p_val:.1f}%</div>
                </div>
                <div style="font-size: 11px; color: #6B7280; margin-top: 2px;">{s_desc}</div>
            </div>
            """
        items_html.append(card_html)
        
    full_html = "".join(items_html)
    st.markdown(full_html, unsafe_allow_html=True)

def render_early_warning_panel(res: dict):
    """
    Renders Early Warning alert box with levels (LOW, MODERATE, HIGH, CRITICAL),
    warning message, time to high risk, and recommended operational action.
    """
    risk = res.get("risk_score", 12.0)
    t_high = res.get("time_to_high_risk")
    t_high_str = f"~{t_high} seconds" if t_high is not None and t_high > 0 else ("CURRENTLY CRITICAL" if risk >= 75 else "N/A (Safe)")
    msg = res.get("warning_message", "Network activity is operating within baseline security parameters.")
    stage = res.get("stage_name", "Normal")
    
    if risk >= 75:
        level_str = "🔴 CRITICAL ALERT"
        box_class = "status-banner-critical"
        action = f"Immediate defensive isolation of host. Terminate abnormal sessions in stage '{stage}'."
    elif risk >= 50:
        level_str = "🟠 HIGH ADVISORY"
        box_class = "status-banner-critical"
        action = f"Escalate monitoring on affected endpoints. Inspect suspicious TCP/UDP flows."
    elif risk >= 25:
        level_str = "🟡 MODERATE WARNING"
        box_class = "status-banner-warning"
        action = "Review anomalous port scan and burst activity. Monitor perimeter firewall logs."
    else:
        level_str = "🟢 LOW RISK (NORMAL)"
        box_class = "status-banner-normal"
        action = "Baseline network activity within normal parameters. Routine SOC monitoring active."

    st.markdown(f"""
    <div class="{box_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 800; font-size: 16px;">{level_str}</span>
            <span style="font-size: 12px; font-weight: 600; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">Time to High Risk: {t_high_str}</span>
        </div>
        <div style="font-size: 14px; margin-bottom: 8px;">{msg}</div>
        <div style="font-size: 12px; opacity: 0.9; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
            <b>Recommended SOC Action:</b> {action}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_live_connection_table(df_connections: pd.DataFrame):
    """
    Renders scrollable live connection table with colored status rows:
    Green -> Normal, Yellow -> Suspicious, Orange -> High Risk, Red -> Critical.
    """
    if df_connections is None or df_connections.empty:
        st.info("No active flows captured yet. Start monitoring to populate the connection table.")
        return

    # Use HTML styling for rich visual table
    table_rows = []
    for _, row in df_connections.iterrows():
        st_val = str(row.get("Status", "Normal")).strip()
        if st_val == "Critical":
            tag_class = "status-critical"
        elif st_val == "High Risk":
            tag_class = "status-highrisk"
        elif st_val == "Suspicious":
            tag_class = "status-suspicious"
        else:
            tag_class = "status-normal"
            
        src_ip = row.get("Source IP", "")
        dst_ip = row.get("Destination IP", "")
        proto = row.get("Protocol", "")
        src_port = row.get("Source Port", 0)
        dst_port = row.get("Destination Port", 0)
        packets = f"{row.get('Packets', 0):,}"
        bytes_val = f"{row.get('Bytes', 0):,}"
        duration = row.get("Duration", "0s")

        tr = (
            f'<tr style="border-bottom: 1px solid #374151; font-size: 12px;">'
            f'<td style="padding: 6px 10px; font-family: monospace;">{src_ip}</td>'
            f'<td style="padding: 6px 10px; font-family: monospace;">{dst_ip}</td>'
            f'<td style="padding: 6px 8px; font-weight: 600;">{proto}</td>'
            f'<td style="padding: 6px 8px; font-family: monospace;">{src_port}</td>'
            f'<td style="padding: 6px 8px; font-family: monospace;">{dst_port}</td>'
            f'<td style="padding: 6px 8px;">{packets}</td>'
            f'<td style="padding: 6px 8px;">{bytes_val}</td>'
            f'<td style="padding: 6px 8px;">{duration}</td>'
            f'<td style="padding: 6px 8px;"><span class="status-tag {tag_class}">{st_val}</span></td>'
            f'</tr>'
        )
        table_rows.append(tr)

    rows_html = "".join(table_rows)
    table_html = (
        '<div class="live-table-container">'
        '<table style="width: 100%; border-collapse: collapse; text-align: left; color: #F9FAFB;">'
        '<thead style="background-color: #1F2937; position: sticky; top: 0; z-index: 1;">'
        '<tr style="border-bottom: 2px solid #4B5563; font-size: 12px; color: #9CA3AF;">'
        '<th style="padding: 8px 10px;">Source IP</th>'
        '<th style="padding: 8px 10px;">Destination IP</th>'
        '<th style="padding: 8px 8px;">Protocol</th>'
        '<th style="padding: 8px 8px;">Src Port</th>'
        '<th style="padding: 8px 8px;">Dst Port</th>'
        '<th style="padding: 8px 8px;">Packets</th>'
        '<th style="padding: 8px 8px;">Bytes</th>'
        '<th style="padding: 8px 8px;">Duration</th>'
        '<th style="padding: 8px 8px;">Status</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


