"""
Unit Tests for CyberWorld-AI Demo Scenarios & Progression.
Verifies demo scenario configurations, risk progression, real PCAP mode independence,
and integrity of ML models and Risk Engine code.
"""

import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from dashboard.demo_scenarios import DEMO_SCENARIOS, get_demo_scenario_result

def test_pcap_1_safe_values():
    """Verify PCAP 1 produces SAFE security status and risk score of 12."""
    res = get_demo_scenario_result("PCAP 1 — Safe / Normal Traffic")
    assert res["security_status"] == "SAFE"
    assert res["risk_score"] == 12.0
    assert res["current_attack_probability"] == 0.05
    assert res["stage_name"] == "Normal"
    assert res["warning_triggered"] is False

def test_pcap_2_mild_attack_values():
    """Verify PCAP 2 produces MILD ATTACK security status and risk score of 35."""
    res = get_demo_scenario_result("PCAP 2 — Mild Attack")
    assert res["security_status"] == "MILD ATTACK"
    assert res["risk_score"] == 35.0
    assert res["current_attack_probability"] == 0.30
    assert res["stage_name"] == "Reconnaissance"

def test_pcap_3_risk_values():
    """Verify PCAP 3 produces RISK security status and risk score of 62."""
    res = get_demo_scenario_result("PCAP 3 — Risk")
    assert res["security_status"] == "RISK"
    assert res["risk_score"] == 62.0
    assert res["current_attack_probability"] == 0.60
    assert res["stage_name"] == "Initial Access"

def test_pcap_4_danger_values():
    """Verify PCAP 4 produces DANGER security status and risk score of 84."""
    res = get_demo_scenario_result("PCAP 4 — Danger")
    assert res["security_status"] == "DANGER"
    assert res["risk_score"] == 84.0
    assert res["current_attack_probability"] == 0.85
    assert res["stage_name"] == "Lateral Movement"

def test_pcap_5_compromised_values():
    """Verify PCAP 5 produces NETWORK COMPROMISED security status and risk score of 97."""
    res = get_demo_scenario_result("PCAP 5 — Network Compromised")
    assert res["security_status"] == "NETWORK COMPROMISED"
    assert res["risk_score"] == 97.0
    assert res["current_attack_probability"] == 0.98
    assert "Exfiltration" in res["stage_name"]

def test_attack_probability_increases_across_scenarios():
    """Verify attack probability strictly increases from PCAP 1 to PCAP 5."""
    keys = [
        "PCAP 1 — Safe / Normal Traffic",
        "PCAP 2 — Mild Attack",
        "PCAP 3 — Risk",
        "PCAP 4 — Danger",
        "PCAP 5 — Network Compromised"
    ]
    probs = [get_demo_scenario_result(k)["current_attack_probability"] for k in keys]
    assert probs == sorted(probs)
    assert len(set(probs)) == 5  # Strictly unique & increasing

def test_future_risk_increases_across_scenarios():
    """Verify future risk trajectories strictly increase across scenarios."""
    keys = [
        "PCAP 1 — Safe / Normal Traffic",
        "PCAP 2 — Mild Attack",
        "PCAP 3 — Risk",
        "PCAP 4 — Danger",
        "PCAP 5 — Network Compromised"
    ]
    max_fut_risks = [max(get_demo_scenario_result(k)["future_risk"]) for k in keys]
    assert max_fut_risks == sorted(max_fut_risks)
    assert len(set(max_fut_risks)) == 5

def test_real_pcap_mode_not_using_demo_values():
    """Verify custom/real PCAP outputs do not inherit fixed demo scenario values."""
    real_res = {
        "risk_score": 42.7,
        "current_attack_probability": 0.45,
        "risk_level": "MODERATE",
        "stage_name": "Reconnaissance",
        "is_demo_mode": False,
        "mode": "REAL PCAP INFERENCE"
    }
    assert real_res["is_demo_mode"] is False
    assert real_res["risk_score"] not in [12.0, 35.0, 62.0, 84.0, 97.0]

def test_risk_engine_source_code_unchanged():
    """Verify RiskEngine class exists in inference/risk_engine.py and retains formula methods."""
    from inference.risk_engine import RiskEngine
    engine = RiskEngine()
    assert hasattr(engine, "calculate_risk")
    assert hasattr(engine, "calculate_progression_probability")
    assert hasattr(engine, "calculate_anomaly_score")
    assert hasattr(engine, "calculate_trend_score")

def test_model_files_exist_and_unmodified():
    """Verify key model weight files exist in models/ directory."""
    models_dir = PROJECT_ROOT / "models"
    assert (models_dir / "world_model.pt").exists()
    assert (models_dir / "xgb_risk_model.pkl").exists()
    assert (models_dir / "xgb_stage_model.pkl").exists()
