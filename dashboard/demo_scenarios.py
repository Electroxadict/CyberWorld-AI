"""
Demo Scenario Definitions & Helpers for CyberWorld-AI Streamlit Control Room Dashboard.
Provides structured scenario simulation values for the 5 built-in demonstration options.
"""

from pathlib import Path

DEMO_SCENARIOS = {
    "PCAP 1 — Safe / Normal Traffic": {
        "pcap_filename": "pcap_1_normal.pcap",
        "scenario_label": "PCAP 1 — Safe / Normal Traffic",
        "security_status": "SAFE",
        "status_level": "NORMAL", # NORMAL, SUSPICIOUS, HIGH RISK, CRITICAL, COMPROMISED
        "risk_score": 12.0,
        "current_attack_probability": 0.05, # 5%
        "risk_level": "LOW",
        "predicted_stage": 0,
        "stage_name": "Normal",
        "warning_triggered": False,
        "warning_priority": "NO WARNING",
        "warning_message": "Network activity is within the normal demonstration baseline.",
        "future_risk": [12.0, 13.0, 14.0, 14.0, 15.0],
        "future_attack_probability": [0.05, 0.06, 0.06, 0.07, 0.07],
        "time_to_high_risk": None,
        "time_to_high_risk_display": "N/A (Safe)",
        "progression_step": 1
    },
    "PCAP 2 — Mild Attack": {
        "pcap_filename": "pcap_2_reconnaissance.pcap",
        "scenario_label": "PCAP 2 — Mild Attack",
        "security_status": "MILD ATTACK",
        "status_level": "SUSPICIOUS",
        "risk_score": 35.0,
        "current_attack_probability": 0.30, # 30%
        "risk_level": "MODERATE",
        "predicted_stage": 1,
        "stage_name": "Reconnaissance",
        "warning_triggered": True,
        "warning_priority": "YES — LOW PRIORITY",
        "warning_message": "Suspicious network activity has been detected in the demonstration scenario.",
        "future_risk": [35.0, 38.0, 41.0, 44.0, 47.0],
        "future_attack_probability": [0.30, 0.34, 0.38, 0.42, 0.46],
        "time_to_high_risk": 20,
        "time_to_high_risk_display": "approximately 20 seconds",
        "progression_step": 2
    },
    "PCAP 3 — Risk": {
        "pcap_filename": "pcap_3_initial_access.pcap",
        "scenario_label": "PCAP 3 — Risk",
        "security_status": "RISK",
        "status_level": "HIGH RISK",
        "risk_score": 62.0,
        "current_attack_probability": 0.60, # 60%
        "risk_level": "HIGH",
        "predicted_stage": 2,
        "stage_name": "Initial Access",
        "warning_triggered": True,
        "warning_priority": "YES — HIGH PRIORITY",
        "warning_message": "Elevated malicious activity indicates increasing security risk.",
        "future_risk": [62.0, 66.0, 70.0, 74.0, 77.0],
        "future_attack_probability": [0.60, 0.65, 0.70, 0.75, 0.79],
        "time_to_high_risk": 10,
        "time_to_high_risk_display": "approximately 10 seconds",
        "progression_step": 3
    },
    "PCAP 4 — Danger": {
        "pcap_filename": "pcap_4_lateral_movement.pcap",
        "scenario_label": "PCAP 4 — Danger",
        "security_status": "DANGER",
        "status_level": "CRITICAL",
        "risk_score": 84.0,
        "current_attack_probability": 0.85, # 85%
        "risk_level": "CRITICAL",
        "predicted_stage": 3,
        "stage_name": "Lateral Movement",
        "warning_triggered": True,
        "warning_priority": "YES — CRITICAL",
        "warning_message": "Critical activity indicates possible active network compromise.",
        "future_risk": [84.0, 87.0, 90.0, 93.0, 95.0],
        "future_attack_probability": [0.85, 0.88, 0.91, 0.94, 0.96],
        "time_to_high_risk": 5,
        "time_to_high_risk_display": "approximately 5 seconds",
        "progression_step": 4
    },
    "PCAP 5 — Network Compromised": {
        "pcap_filename": "pcap_5_exfiltration.pcap",
        "scenario_label": "PCAP 5 — Network Compromised",
        "security_status": "NETWORK COMPROMISED",
        "status_level": "COMPROMISED",
        "risk_score": 97.0,
        "current_attack_probability": 0.98, # 98%
        "risk_level": "CRITICAL",
        "predicted_stage": 5,
        "stage_name": "Data Exfiltration / Compromise",
        "warning_triggered": True,
        "warning_priority": "YES — CRITICAL",
        "warning_message": "Severe malicious activity indicates that the network should be treated as compromised.",
        "future_risk": [97.0, 98.0, 99.0, 99.0, 100.0],
        "future_attack_probability": [0.98, 0.98, 0.99, 0.99, 1.00],
        "time_to_high_risk": 0,
        "time_to_high_risk_display": "IMMEDIATE",
        "progression_step": 5
    }
}

def get_demo_scenario_result(scenario_key: str, pipeline_real_res: dict = None) -> dict:
    """
    Constructs prediction result dictionary for Demo Mode.
    Combines scenario demonstration values for Risk & Trajectory with actual
    model-generated SHAP and Temporal Attention from the ML pipeline (Option A).
    """
    cfg = DEMO_SCENARIOS.get(scenario_key)
    if not cfg:
        return None
        
    res = dict(pipeline_real_res) if pipeline_real_res else {}
    
    res["mode"] = "DEMO MODE — SCENARIO SIMULATION"
    res["is_demo_mode"] = True
    res["pcap_file"] = cfg["pcap_filename"]
    res["scenario_name"] = cfg["scenario_label"]
    res["security_status"] = cfg["security_status"]
    res["status_level"] = cfg["status_level"]
    res["risk_score"] = cfg["risk_score"]
    res["current_attack_probability"] = cfg["current_attack_probability"]
    res["risk_level"] = cfg["risk_level"]
    res["predicted_stage"] = cfg["predicted_stage"]
    res["stage_name"] = cfg["stage_name"]
    res["warning_triggered"] = cfg["warning_triggered"]
    res["warning_priority"] = cfg["warning_priority"]
    res["warning_message"] = cfg["warning_message"]
    res["future_risk"] = cfg["future_risk"]
    res["future_attack_probability"] = cfg["future_attack_probability"]
    res["time_to_high_risk"] = cfg["time_to_high_risk"]
    res["time_to_high_risk_display"] = cfg["time_to_high_risk_display"]
    res["progression_step"] = cfg["progression_step"]
    
    if "flow_count" not in res:
        res["flow_count"] = 120
    if "window_count" not in res:
        res["window_count"] = 12
    if "top_shap_features" not in res:
        res["top_shap_features"] = [
            {"feature": "fut_pct_ACK Flag Cnt_std", "shap_value": 0.45},
            {"feature": "curr_SYN Flag Cnt_mean", "shap_value": 0.32},
            {"feature": "fut_slope_Src Port_std", "shap_value": 0.28}
        ]
    if "attention_weights" not in res:
        res["attention_weights"] = [0.08, 0.09, 0.08, 0.10, 0.11, 0.12, 0.15, 0.11, 0.09, 0.07]
    if "most_influential_timestep" not in res:
        res["most_influential_timestep"] = "t-3"
    if "temporal_explanation" not in res:
        res["temporal_explanation"] = "The World Model placed the highest temporal attention on window 't-3', indicating historical traffic patterns influenced the forecast."
    if "shap_explanation" not in res:
        res["shap_explanation"] = f"Demonstration Threat Evaluation: {cfg['security_status']} (Risk Score: {cfg['risk_score']:.0f}/100).\nTop Influential Feature Drivers: fut_pct_ACK Flag Cnt_std, curr_SYN Flag Cnt_mean."
        
    return res
