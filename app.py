"""
CyberWorld-AI — Streamlit Predictive Cyber Defence Control Room Dashboard.
Launch with: streamlit run app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
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
    render_pipeline_flow,
    render_model_info_expander
)
from dashboard.charts import (
    create_risk_gauge_chart,
    create_future_risk_chart,
    create_attack_prob_chart,
    create_shap_bar_chart,
    create_shap_group_chart,
    create_attention_chart
)
from attack_mapping.mitre_mapper import MitreStageMapper
from scripts.generate_sample_pcap import generate_sample_pcap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="CyberWorld-AI SOC Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_dark_soc_css()
    render_header()
    
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    render_status_bar(device_name=device_name)
    
    # Load ML Pipeline (cached)
    try:
        pipeline = load_pipeline_cached()
        mitre_mapper = MitreStageMapper()
    except Exception as e:
        st.error(f"Failed to initialize CyberWorld-AI Predictive Defence Pipeline: {e}")
        st.stop()
        
    # --- Sidebar Controls ---
    st.sidebar.title("🛡️ Defence Controls")
    
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_pcap_path = data_dir / "sample_test.pcap"
    if not sample_pcap_path.exists():
        try:
            generate_sample_pcap(sample_pcap_path)
        except Exception:
            pass
            
    pcap_source = st.sidebar.radio(
        "Select PCAP Input Source:",
        ["Select Sample / Existing PCAP", "Upload PCAP File (.pcap / .pcapng)"]
    )
    
    target_pcap_path = None
    is_synthetic_sample = False
    
    if pcap_source == "Select Sample / Existing PCAP":
        existing_files = list(data_dir.glob("*.pcap")) + list(data_dir.glob("*.pcapng"))
        file_names = [f.name for f in existing_files]
        
        if file_names:
            selected_name = st.sidebar.selectbox("Choose PCAP File:", file_names)
            target_pcap_path = data_dir / selected_name
            if selected_name == "sample_test.pcap":
                is_synthetic_sample = True
                st.sidebar.warning("⚠️ Selected file is a SYNTHETIC TEST FIXTURE generated via Scapy.")
        else:
            st.sidebar.error("No existing .pcap files found in data/raw/.")
            
    else:
        uploaded_file = st.sidebar.file_uploader("Upload .pcap or .pcapng file", type=["pcap", "pcapng"])
        if uploaded_file is not None:
            uploaded_save_path = data_dir / f"uploaded_{uploaded_file.name}"
            with open(uploaded_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_pcap_path = uploaded_save_path
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Analysis Configuration:**")
    st.sidebar.text("Prediction Horizon: 5 steps (+25s)")
    st.sidebar.text("Time Window: 5 seconds")
    
    analyze_btn = st.sidebar.button("🚀 Analyze PCAP Traffic", use_container_width=True)
    
    # Store prediction result in session state
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
        
    if analyze_btn and target_pcap_path:
        if not target_pcap_path.exists():
            st.error(f"Target PCAP file does not exist: {target_pcap_path}")
        else:
            with st.spinner(f"Parsing PCAP traffic & executing World Model simulation for {target_pcap_path.name}..."):
                try:
                    res = pipeline.predict(target_pcap_path)
                    st.session_state["prediction_result"] = res
                    st.session_state["pcap_name"] = target_pcap_path.name
                    st.session_state["is_synthetic"] = is_synthetic_sample
                    st.success("Analysis Complete!")
                except Exception as err:
                    st.error(f"PCAP Predictive Pipeline Analysis Failed: {err}")
                    with st.expander("Technical Details"):
                        st.exception(err)
                        
    # If no analysis run yet, auto-run on default sample PCAP if available
    if st.session_state["prediction_result"] is None and sample_pcap_path.exists():
        try:
            res = pipeline.predict(sample_pcap_path)
            st.session_state["prediction_result"] = res
            st.session_state["pcap_name"] = sample_pcap_path.name
            st.session_state["is_synthetic"] = True
        except Exception:
            pass
            
    res = st.session_state.get("prediction_result")
    
    if res is None:
        st.info("👈 Please select or upload a .pcap file in the sidebar and click **Analyze PCAP Traffic**.")
        return
        
    if st.session_state.get("is_synthetic", False):
        st.warning("⚠️ **NOTICE**: Currently analyzing `sample_test.pcap` — SYNTHETIC TEST FIXTURE generated via Scapy for local testing. Results reflect synthetic flow attributes.")
        
    # Render Top KPI Cards & Security Status Panel
    render_main_status_panel(res)
    render_kpi_cards(res)
    st.markdown("---")
    
    # --- 5 Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 1. Overview",
        "🔮 2. Future Prediction & Trajectory",
        "💡 3. Explainability (SHAP & Attention)",
        "🌐 4. Network Telemetry & Flows",
        "⚙️ 5. System & Model Info"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.subheader("Security Risk Overview & Multi-Step Forecast")
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            fig_gauge = create_risk_gauge_chart(res["risk_score"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_g2:
            fig_fut_risk = create_future_risk_chart(res["future_risk"])
            st.plotly_chart(fig_fut_risk, use_container_width=True)
            
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            fig_prob = create_attack_prob_chart(res["current_attack_probability"], res["future_attack_probability"])
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with col_c2:
            st.markdown("### 📋 PCAP Traffic Summary")
            st.markdown(f"**PCAP File**: `{res['pcap_file']}`")
            st.markdown(f"**Total Extracted Flows**: `{res['flow_count']}`")
            st.markdown(f"**Temporal 5s Windows**: `{res['window_count']}`")
            st.markdown(f"**Analyzed Sequence Window**: `10 states (50 seconds)`")
            st.markdown(f"**Current Risk Score**: `{res['risk_score']:.1f} / 100` (`{res['risk_level']}`)")
            st.markdown(f"**Predicted Attack Stage**: `{res['stage_name']}`")
            st.markdown(f"**Early Warning Advisory**: {res['warning_message']}")

    # --- TAB 2: FUTURE PREDICTION ---
    with tab2:
        st.subheader("PyTorch World Model 5-Step Recursive Rollout Simulation")
        
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("### 📈 Predicted Risk & Attack Trajectory")
            traj_df = pd.DataFrame({
                "Simulation Step": ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"],
                "Risk Score (0-100)": [res["risk_score"]] + list(res["future_risk"]),
                "Attack Probability (%)": [res["current_attack_probability"] * 100.0] + [p * 100.0 for p in res["future_attack_probability"]]
            })
            st.dataframe(traj_df, use_container_width=True)
            
        with col_p2:
            st.markdown("### 🎯 Predicted MITRE ATT&CK Stage")
            stage_info = mitre_mapper.explain_mapping(res["predicted_stage"])
            st.markdown(f"**Current Stage ID**: `{res['predicted_stage']}`")
            st.markdown(f"**Stage Name**: `{stage_info['stage_name']}`")
            st.markdown(f"**Technical Reason**: {stage_info['reason']}")
            st.info(f"**Description**: {stage_info['description']}")
            
        st.markdown("---")
        st.markdown("### ⚠️ Early Warning Alert Assessment")
        if res["warning_triggered"]:
            st.error(f"⚠️ **ALERT TRIGGERED**: {res['warning_message']}")
            if res['time_to_high_risk'] is not None:
                st.markdown(f"**Estimated Time to High Risk Threshold**: `{res['time_to_high_risk']} seconds`")
        else:
            st.success(f"✓ **NORMAL**: {res['warning_message']}")

    # --- TAB 3: EXPLAINABILITY ---
    with tab3:
        st.subheader("Explainable AI: TreeSHAP Contributions & World Model Temporal Attention")
        
        col_ex_opt1, col_ex_opt2 = st.columns([1, 1])
        with col_ex_opt1:
            top_n_choice = st.radio("SHAP Feature Count:", [10, 20], horizontal=True)
            
        fig_shap = create_shap_bar_chart(res["top_shap_features"], top_n=top_n_choice)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.markdown("### 📝 Human-Readable SHAP Threat Explanation")
        st.code(res.get("shap_explanation", "N/A"), language="text")
        
        col_sh_grp, col_sh_att = st.columns([1, 1])
        with col_sh_grp:
            group_imp_data = res.get("feature_group_importance", {})
            if isinstance(group_imp_data, dict) and group_imp_data:
                fig_grp = create_shap_group_chart(group_imp_data)
                st.plotly_chart(fig_grp, use_container_width=True)
            else:
                st.info("Feature group SHAP data unavailable.")
            
        with col_sh_att:
            fig_attn = create_attention_chart(
                ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"],
                res["attention_weights"],
                res["most_influential_timestep"]
            )
            st.plotly_chart(fig_attn, use_container_width=True)
            
        st.markdown("### ⏳ Temporal Window Attention Explanation")
        st.info(res["temporal_explanation"])

    # --- TAB 4: NETWORK TELEMETRY ---
    with tab4:
        st.subheader("Extracted Network Telemetry & Flow Records")
        
        st.markdown(f"### 🔍 Extracted Flows Table (`{res['flow_count']}` total flows)")
        # Show flow table if available via pipeline
        st.info(f"Analyzed {res['flow_count']} flows aggregated into {res['window_count']} 5-second temporal state windows.")
        
        st.markdown("### 📊 69-Feature Current State Vector")
        top_shap_df = pd.DataFrame(res["top_shap_features"])
        st.dataframe(top_shap_df, use_container_width=True)

    # --- TAB 5: SYSTEM INFO & EXPORT ---
    with tab5:
        st.subheader("System Architecture & Execution Status")
        render_pipeline_flow()
        render_model_info_expander(pipeline.config)
        
        st.markdown("---")
        st.markdown("### 💾 Export Structured Prediction Analysis")
        json_data = json.dumps(res, indent=2)
        st.download_button(
            label="📥 Download Analysis Report (JSON)",
            data=json_data,
            file_name=f"cyberworld_analysis_{res['pcap_file']}.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
