"""
CyberWorld-AI — Real-Time AI Security Operations Center (SOC) Control Room Dashboard.
Monitors live network traffic from Wi-Fi, Ethernet, and Loopback interfaces,
predicts cyber threats using PyTorch LSTM World Model recursive rollouts (+25s),
classifies attack progression with XGBoost, computes quantitative risk via Risk Engine,
explains predictions via TreeSHAP and Temporal Attention, and provides continuous live telemetry.

Launch with: streamlit run app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import time
import json
import logging
import yaml
import pandas as pd
import streamlit as st
import torch

from dashboard.components import (
    load_pipeline_cached,
    inject_dark_soc_css,
    render_header,
    render_status_bar,
    render_kpi_cards,
    render_main_status_panel,
    render_mode_indicator,
    render_progression_stepper,
    render_threat_level_progression,
    render_responsive_kpi_row,
    render_mitre_panel,
    render_early_warning_panel,
    render_live_connection_table,
    render_pipeline_flow,
    render_model_info_expander,
    render_inference_debug_info
)
from dashboard.charts import (
    create_risk_gauge_chart,
    create_future_risk_chart,
    create_attack_prob_chart,
    create_shap_bar_chart,
    create_shap_group_chart,
    create_attention_chart,
    create_pps_chart,
    create_active_connections_chart,
    create_bandwidth_chart,
    create_risk_trend_chart,
    create_attack_prob_trend_chart,
    create_protocol_distribution_chart,
    create_top_dst_ips_chart,
    create_top_src_ips_chart,
    create_topology_chart
)
from dashboard.demo_scenarios import DEMO_SCENARIOS, get_demo_scenario_result
from attack_mapping.mitre_mapper import MitreStageMapper
from scripts.generate_sih_demo_pcaps import generate_all_sih_pcaps
from preprocessing.live_capture import get_live_capture_engine, get_available_interfaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_config_settings(new_settings: dict):
    """Persists user configurable settings to config.yaml."""
    cfg_path = PROJECT_ROOT / "config.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
        if "live_monitoring" not in cfg:
            cfg["live_monitoring"] = {}
        if "temporal" not in cfg:
            cfg["temporal"] = {}
            
        cfg["live_monitoring"]["interface"] = new_settings.get("interface", "Wi-Fi")
        cfg["live_monitoring"]["refresh_interval"] = int(new_settings.get("refresh_interval", 2))
        cfg["live_monitoring"]["window_duration"] = int(new_settings.get("window_duration", 5))
        cfg["live_monitoring"]["max_buffer_packets"] = int(new_settings.get("max_buffer_packets", 50000))
        cfg["live_monitoring"]["auto_export"] = bool(new_settings.get("auto_export", False))
        cfg["temporal"]["prediction_horizon"] = int(new_settings.get("prediction_horizon", 5))
        
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Failed to update config.yaml: {e}")
        return False

def generate_csv_exports(res: dict, df_conn: pd.DataFrame, df_rolling: pd.DataFrame) -> dict:
    """Generates downloadable CSV strings for Flow Table, Feature Table, and Risk History."""
    csvs = {}
    
    # 1. Flow Table CSV
    if df_conn is not None and not df_conn.empty:
        csvs["flow_table.csv"] = df_conn.to_csv(index=False)
    else:
        csvs["flow_table.csv"] = "Source IP,Destination IP,Protocol,Source Port,Destination Port,Packets,Bytes,Duration,Status\n"
        
    # 2. Feature Table CSV (SHAP Top Features)
    top_feats = res.get("top_shap_features", [])
    if top_feats:
        df_feats = pd.DataFrame(top_feats)
        csvs["feature_importance_table.csv"] = df_feats.to_csv(index=False)
    else:
        csvs["feature_importance_table.csv"] = "feature,value,shap_value,direction,importance\n"
        
    # 3. Risk History CSV
    if df_rolling is not None and not df_rolling.empty:
        csvs["risk_history.csv"] = df_rolling.to_csv(index=False)
    else:
        csvs["risk_history.csv"] = "timestamp,epoch,pps,mbps,active_connections,risk_score,attack_probability\n"
        
    return csvs

def main():
    st.set_page_config(
        page_title="CyberWorld-AI SOC Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_dark_soc_css()
    render_header()
    
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    render_status_bar(device_name=device_name)
    
    # Load ML Pipeline (cached)
    try:
        pipeline = load_pipeline_cached()
        mitre_mapper = MitreStageMapper()
    except Exception as e:
        st.error(f"Failed to initialize CyberWorld-AI Predictive Defence Pipeline: {e}")
        st.stop()
        
    # Initialize Live Capture Engine
    live_engine = get_live_capture_engine()
    available_ifaces = get_available_interfaces()
    
    # Ensure SIH Demo PCAPs exist
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pcap_1 = data_dir / "pcap_1_normal.pcap"
    if not pcap_1.exists():
        try:
            generate_all_sih_pcaps(data_dir)
        except Exception:
            pass

    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("🛡️ SOC Operations Control")
    
    data_source = st.sidebar.radio(
        "DATA SOURCE",
        [
            "⚡ Live Monitoring",
            "📁 Existing PCAP",
            "📤 Upload PCAP",
            "🧪 Demo Mode"
        ],
        index=0
    )
    
    target_pcap_path = None
    demo_scenario_choice = None
    
    if data_source == "⚡ Live Monitoring":
        st.sidebar.markdown("### 🌐 INTERFACE")
        
        # Interface dropdown
        def_idx = 0
        for idx, if_name in enumerate(available_ifaces):
            if "wi-fi" in if_name.lower():
                def_idx = idx
                break
                
        selected_iface = st.sidebar.selectbox("Select Network Interface:", available_ifaces, index=def_idx)
        live_engine.selected_interface = selected_iface
        
        # Start / Stop Buttons
        col_btn1, col_btn2 = st.sidebar.columns(2)
        with col_btn1:
            if st.button("▶ Start", use_container_width=True, type="primary", disabled=live_engine.is_running):
                live_engine.start(selected_iface)
                st.session_state["live_monitoring_active"] = True
                st.rerun()
                
        with col_btn2:
            if st.button("⏹ Stop", use_container_width=True, disabled=not live_engine.is_running):
                live_engine.stop()
                st.session_state["live_monitoring_active"] = False
                st.rerun()

        # Live Status Indicator
        if live_engine.is_running:
            st.sidebar.markdown(f"""
            <div style="margin: 12px 0;">
                <span class="live-badge-active">● LIVE MONITORING</span>
                <div style="font-size: 11px; color: #10B981; margin-top: 4px;">{live_engine.status_message}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style="margin: 12px 0;">
                <span class="live-badge-stopped">■ STOPPED</span>
                <div style="font-size: 11px; color: #EF4444; margin-top: 4px;">Capture inactive. Click Start to sniff live traffic.</div>
            </div>
            """, unsafe_allow_html=True)
            
        auto_refresh = st.sidebar.checkbox("Auto-Refresh Dashboard (Live)", value=live_engine.is_running)
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds):", min_value=1, max_value=10, value=2)

    elif data_source == "📁 Existing PCAP":
        st.sidebar.markdown("### 📁 PCAP FILE")
        existing_files = sorted(list(data_dir.glob("*.pcap")) + list(data_dir.glob("*.pcapng")))
        file_names = [f.name for f in existing_files]
        if file_names:
            selected_name = st.sidebar.selectbox("Choose PCAP File:", file_names)
            target_pcap_path = data_dir / selected_name
        else:
            st.sidebar.error("No .pcap files found in data/raw/.")
            
    elif data_source == "📤 Upload PCAP":
        st.sidebar.markdown("### 📤 UPLOAD PCAP")
        uploaded_file = st.sidebar.file_uploader("Upload .pcap / .pcapng file", type=["pcap", "pcapng"])
        if uploaded_file is not None:
            uploaded_save_path = data_dir / f"uploaded_{uploaded_file.name}"
            with open(uploaded_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_pcap_path = uploaded_save_path
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            
    else: # 🧪 Demo Mode
        st.sidebar.markdown("### 🧪 SIH DEMO SCENARIOS")
        demo_scenario_choice = st.sidebar.radio(
            "Select Demonstration Scenario:",
            [
                "PCAP 1 — Safe / Normal Traffic",
                "PCAP 2 — Mild Attack",
                "PCAP 3 — Risk",
                "PCAP 4 — Danger",
                "PCAP 5 — Network Compromised"
            ]
        )
        if demo_scenario_choice in DEMO_SCENARIOS:
            target_pcap_path = data_dir / DEMO_SCENARIOS[demo_scenario_choice]["pcap_filename"]

    st.sidebar.markdown("---")
    
    # --- PIPELINE INFERENCE EXECUTION ---
    res = None
    live_metrics = live_engine.get_live_metrics()
    df_conn = live_engine.get_connection_table()
    df_rolling = live_engine.get_rolling_history()
    proto_dist = live_engine.get_protocol_distribution()
    top_src, top_dst = live_engine.get_top_talkers()
    topo_data = live_engine.get_topology_data()
    
    if data_source == "⚡ Live Monitoring":
        # Extract live window sequence if available
        df_seq = live_engine.get_latest_sequence_dataframe()
        if not df_seq.empty and len(df_seq) >= 10:
            try:
                res = pipeline.predict_windows(
                    df_windows=df_seq,
                    source_name="Live Network Monitoring",
                    identifier=live_engine.selected_interface,
                    flow_count=live_metrics["total_flows"]
                )
                res["mode"] = f"LIVE NETWORK MONITORING ({live_engine.selected_interface})"
                res["is_demo_mode"] = False
                res["security_status"] = res["risk_level"]
                
                # Feed live calculated risk into rolling history
                with live_engine._lock:
                    if live_engine.rolling_telemetry:
                        live_engine.rolling_telemetry[-1]["risk_score"] = float(res["risk_score"])
                        live_engine.rolling_telemetry[-1]["attack_probability"] = float(res["current_attack_probability"])
                        
                st.session_state["prediction_result"] = res
            except Exception as err:
                logger.debug(f"Live prediction cycle: {err}")
                res = st.session_state.get("prediction_result")
        else:
            res = st.session_state.get("prediction_result")
            
        if res is None:
            # Baseline quiet system state until first window batch accumulates
            res = {
                "source": "Live Network Monitoring",
                "pcap_file": live_engine.selected_interface,
                "flow_count": live_metrics["total_flows"],
                "window_count": max(1, live_metrics["completed_windows_count"]),
                "current_attack_probability": 0.05,
                "risk_score": 12.0,
                "risk_level": "LOW",
                "future_risk": [12.0, 13.0, 14.0, 15.0, 15.0],
                "future_attack_probability": [0.05, 0.06, 0.06, 0.07, 0.07],
                "predicted_stage": 0,
                "stage_name": "Normal",
                "stage_probabilities": [0.94, 0.02, 0.01, 0.01, 0.01, 0.01],
                "time_to_high_risk": None,
                "warning_triggered": False,
                "warning_message": "Network activity is within normal baseline operational parameters.",
                "top_shap_features": [
                    {"feature": "TCP Win Mean_mean", "value": 64240.0, "shap_value": -0.15, "direction": "DECREASES_RISK", "importance": 0.15},
                    {"feature": "SYN_ACK_Ratio", "value": 0.1, "shap_value": -0.12, "direction": "DECREASES_RISK", "importance": 0.12},
                    {"feature": "Port_Diversity", "value": 0.25, "shap_value": -0.08, "direction": "DECREASES_RISK", "importance": 0.08}
                ],
                "feature_group_importance": {
                    "CURRENT": {"percentage_share": 35.0},
                    "FUTURE MEAN": {"percentage_share": 25.0},
                    "WORLD MODEL THREAT FEATURES": {"percentage_share": 40.0}
                },
                "shap_explanation": "Top feature drivers reflect balanced TCP window parameters and normal port diversity.",
                "attention_weights": [0.1] * 10,
                "most_influential_timestep": "t",
                "most_influential_weight": 0.15,
                "temporal_explanation": "Temporal attention is evenly distributed across baseline traffic windows.",
                "mode": f"LIVE MONITORING ({live_engine.selected_interface})",
                "is_demo_mode": False,
                "security_status": "SAFE",
                "inference_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time_seconds": 0.01
            }
            st.session_state["prediction_result"] = res

    else:
        # PCAP or Demo Mode Inference
        current_key = f"{data_source}_{target_pcap_path.name if target_pcap_path else ''}_{demo_scenario_choice or ''}"
        last_key = st.session_state.get("last_analyzed_key")
        
        if target_pcap_path and target_pcap_path.exists() and (current_key != last_key or st.session_state.get("prediction_result") is None):
            with st.spinner(f"Analyzing {target_pcap_path.name}..."):
                try:
                    real_pipeline_res = pipeline.predict(target_pcap_path)
                    
                    if data_source == "🧪 Demo Mode":
                        res = get_demo_scenario_result(demo_scenario_choice, real_pipeline_res)
                    else:
                        res = real_pipeline_res
                        res["mode"] = "REAL PCAP INFERENCE"
                        res["is_demo_mode"] = False
                        res["security_status"] = res["risk_level"]
                        
                    st.session_state["prediction_result"] = res
                    st.session_state["last_analyzed_key"] = current_key
                except Exception as err:
                    st.error(f"Inference Error: {err}")
                    
        res = st.session_state.get("prediction_result")

    if res is None:
        st.info("👈 Please select a Data Source or Start Live Monitoring in the sidebar.")
        return

    # --- SIDEBAR EXPORT BUTTONS ---
    st.sidebar.markdown("### 💾 EXPORT TELEMETRY")
    
    # 1. JSON Export
    json_export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": res.get("source", "SOC"),
        "connections": live_metrics["active_connections"],
        "flows": live_metrics["total_flows"],
        "risk_score": res["risk_score"],
        "risk_level": res["risk_level"],
        "mitre_stage": {
            "id": res["predicted_stage"],
            "name": res["stage_name"],
            "probabilities": res.get("stage_probabilities", [])
        },
        "future_prediction": {
            "steps": ["+5s", "+10s", "+15s", "+20s", "+25s"],
            "risk": res["future_risk"],
            "probability": res["future_attack_probability"]
        },
        "shap": res["top_shap_features"],
        "attention": {
            "weights": res["attention_weights"],
            "most_influential": res["most_influential_timestep"]
        }
    }
    
    st.sidebar.download_button(
        label="📥 Export JSON",
        data=json.dumps(json_export_data, indent=2),
        file_name=f"cyberworld_soc_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # 2. CSV Exports
    csv_dict = generate_csv_exports(res, df_conn, df_rolling)
    st.sidebar.download_button(
        label="📊 Export CSV (Flows)",
        data=csv_dict["flow_table.csv"],
        file_name=f"cyberworld_flows_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # --- TOP CONTROL ROOM PANELS ---
    render_mode_indicator(res)
    if res.get("is_demo_mode", False):
        render_progression_stepper(res)
    else:
        # Real Threat Level Horizontal Progression
        render_threat_level_progression(res["risk_score"])
        
    render_main_status_panel(res)
    
    # Responsive Top KPI Row (8 cards)
    render_responsive_kpi_row(live_metrics, res)
    st.markdown("---")

    # --- 5 SOC CONTROL ROOM TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛡️ 1. Live SOC Overview",
        "🔮 2. Future Rollout & MITRE ATT&CK",
        "💡 3. Explainability (SHAP & Attention)",
        "🌐 4. Live Telemetry & Connections",
        "⚙️ 5. Settings & System Lineage"
    ])

    # ==========================================
    # TAB 1: LIVE SOC OVERVIEW
    # ==========================================
    with tab1:
        st.subheader("Security Posture & Early Warning Advisory")
        
        # Early Warning Panel
        render_early_warning_panel(res)
        
        col_ov1, col_ov2 = st.columns([1, 1])
        with col_ov1:
            fig_gauge = create_risk_gauge_chart(res["risk_score"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_ov2:
            fig_fut = create_future_risk_chart(res["future_risk"])
            st.plotly_chart(fig_fut, use_container_width=True)
            
        col_ov3, col_ov4 = st.columns([1, 1])
        with col_ov3:
            fig_prob = create_attack_prob_chart(res["current_attack_probability"], res["future_attack_probability"])
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with col_ov4:
            st.markdown("### 🕸️ Live Network Node Topology")
            fig_topo = create_topology_chart(topo_data)
            st.plotly_chart(fig_topo, use_container_width=True)

    # ==========================================
    # TAB 2: FUTURE ROLLOUT & MITRE ATT&CK
    # ==========================================
    with tab2:
        st.subheader("PyTorch World Model 5-Step Recursive Rollout Forecast (+25s Horizon)")
        
        col_ro1, col_ro2 = st.columns([1, 1])
        with col_ro1:
            st.markdown("### 📈 Projected Multi-Step Threat Trajectory")
            traj_df = pd.DataFrame({
                "Forecast Horizon": ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"],
                "Predicted Risk Score (0-100)": [res["risk_score"]] + list(res["future_risk"]),
                "Attack Probability (%)": [res["current_attack_probability"] * 100.0] + [p * 100.0 for p in res["future_attack_probability"]]
            })
            st.dataframe(traj_df, use_container_width=True)
            
            # Additional trend chart
            fig_atk_trend = create_attack_prob_trend_chart(df_rolling)
            st.plotly_chart(fig_atk_trend, use_container_width=True)
            
        with col_ro2:
            st.markdown("### 🎯 MITRE ATT&CK Stage Classification Matrix")
            stage_info = mitre_mapper.explain_mapping(res["predicted_stage"])
            st.info(f"**Current Classification**: Stage {res['predicted_stage']} — `{res['stage_name']}`\n\n**Description**: {stage_info.get('description', '')}")
            
            # MITRE ATT&CK Panel with active highlight and probabilities
            render_mitre_panel(res["predicted_stage"], res.get("stage_probabilities"))

    # ==========================================
    # TAB 3: EXPLAINABILITY (SHAP & ATTENTION)
    # ==========================================
    with tab3:
        st.subheader("Explainable AI: TreeSHAP Feature Drivers & PyTorch Temporal Attention")
        
        col_exp1, col_exp2 = st.columns([1, 1])
        with col_exp1:
            st.markdown("### 🔍 Top 10 Feature Contributions (TreeSHAP)")
            fig_shap = create_shap_bar_chart(res["top_shap_features"], top_n=10)
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.markdown("**Natural Language Threat Explanation:**")
            st.code(res.get("shap_explanation", "Normal baseline operations."), language="text")
            
        with col_exp2:
            st.markdown("### ⏳ Temporal Window Attention Weights ($t-9 \\dots t$)")
            fig_attn = create_attention_chart(
                ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"],
                res["attention_weights"],
                res["most_influential_timestep"]
            )
            st.plotly_chart(fig_attn, use_container_width=True)
            
            st.markdown(f"**Most Influential Timestep**: `{res['most_influential_timestep']}` (Weight: `{res['most_influential_weight']*100:.1f}%`)")
            st.info(res.get("temporal_explanation", "Attention distributed evenly."))

    # ==========================================
    # TAB 4: LIVE TELEMETRY & CONNECTIONS
    # ==========================================
    with tab4:
        st.subheader("Live Network Telemetry & Connection Monitor")
        
        st.markdown("### 📋 Active Network Connection Table")
        render_live_connection_table(df_conn)
        st.markdown("---")
        
        st.markdown("### 📊 Real-Time Rolling Telemetry Charts (Last 5 Minutes)")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(create_pps_chart(df_rolling), use_container_width=True)
            st.plotly_chart(create_bandwidth_chart(df_rolling), use_container_width=True)
            st.plotly_chart(create_top_src_ips_chart(top_src), use_container_width=True)
            
        with col_c2:
            st.plotly_chart(create_active_connections_chart(df_rolling), use_container_width=True)
            st.plotly_chart(create_risk_trend_chart(df_rolling), use_container_width=True)
            st.plotly_chart(create_top_dst_ips_chart(top_dst), use_container_width=True)
            
        st.plotly_chart(create_protocol_distribution_chart(proto_dist), use_container_width=True)

    # ==========================================
    # TAB 5: SETTINGS & SYSTEM LINEAGE
    # ==========================================
    with tab5:
        st.subheader("SOC Configuration & System Lineage")
        
        col_set1, col_set2 = st.columns([1, 1])
        with col_set1:
            st.markdown("### ⚙️ Operational Settings")
            with st.form("settings_form"):
                set_iface = st.selectbox("Default Network Interface:", available_ifaces, index=0)
                set_win_dur = st.number_input("Time Window Duration (seconds):", min_value=1, max_value=60, value=5)
                set_ref_int = st.number_input("Dashboard Refresh Interval (seconds):", min_value=1, max_value=30, value=2)
                set_horizon = st.number_input("Prediction Rollout Horizon (steps):", min_value=1, max_value=10, value=5)
                set_max_pkts = st.number_input("Max Packet Rolling Buffer:", min_value=1000, max_value=500000, value=50000, step=5000)
                set_auto_exp = st.checkbox("Enable Automatic Export on Threat Escalation", value=False)
                
                submitted = st.form_submit_button("💾 Save Settings to config.yaml", use_container_width=True)
                if submitted:
                    success = save_config_settings({
                        "interface": set_iface,
                        "window_duration": set_win_dur,
                        "refresh_interval": set_ref_int,
                        "prediction_horizon": set_horizon,
                        "max_buffer_packets": set_max_pkts,
                        "auto_export": set_auto_exp
                    })
                    if success:
                        st.success("Configuration updated and saved to config.yaml!")
                    else:
                        st.error("Failed to save configuration.")
                        
            st.markdown("### 📁 Batch CSV Telemetry Exports")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.download_button("📥 Flows (CSV)", csv_dict["flow_table.csv"], f"flows_{int(time.time())}.csv", "text/csv", use_container_width=True)
            with col_d2:
                st.download_button("📥 SHAP Features (CSV)", csv_dict["feature_importance_table.csv"], f"shap_{int(time.time())}.csv", "text/csv", use_container_width=True)
            with col_d3:
                st.download_button("📥 Risk Trend (CSV)", csv_dict["risk_history.csv"], f"risk_trend_{int(time.time())}.csv", "text/csv", use_container_width=True)
                
        with col_set2:
            st.markdown("### 🏛️ Pipeline Architecture & Lineage")
            render_pipeline_flow()
            render_model_info_expander(pipeline.config)
            render_inference_debug_info(res, target_pcap_path)

    # --- AUTO-REFRESH LOOP FOR LIVE MONITORING ---
    if data_source == "⚡ Live Monitoring" and live_engine.is_running and auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
