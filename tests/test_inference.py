"""
Unit and Integration Tests for CyberWorld-AI Inference, Rollout, Risk Engine, and Early Warning System.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import WorldModelPredictor
from inference.rollout import RolloutSimulator
from inference.risk_engine import RiskEngine
from inference.early_warning import EarlyWarningEngine

def test_risk_engine_calculations():
    risk_engine = RiskEngine()
    
    # 1. Risk level boundaries (0-50 LOW, 50-75 MODERATE, 75-90 HIGH, 90-100 CRITICAL)
    assert risk_engine.classify_risk_level(10) == "LOW"
    assert risk_engine.classify_risk_level(55) == "MODERATE"
    assert risk_engine.classify_risk_level(80) == "HIGH"
    assert risk_engine.classify_risk_level(95) == "CRITICAL"
    
    # 2. Progression probability (max of future probs)
    fut_probs = np.array([0.2, 0.4, 0.85, 0.6])
    prog_p = risk_engine.calculate_progression_probability(fut_probs)
    assert pytest.approx(prog_p, 0.01) == 0.85
    
    # 3. Trend score bounds
    trend = risk_engine.calculate_trend_score(0.2, fut_probs)
    assert 0.0 <= trend <= 1.0
    
    # 4. Overall risk clamping (0 to 100)
    risk_res = risk_engine.calculate_risk(1.2, np.array([1.5, 2.0]))
    assert 0.0 <= risk_res["risk_score"] <= 100.0

def test_early_warning_logic():
    ew_engine = EarlyWarningEngine()

    # 1. Stage decoding
    assert ew_engine.decode_stage_name(0) == "Normal"
    assert ew_engine.decode_stage_name(1) == "Reconnaissance"

    # 2. Time-to-high-risk estimation (5s window size)
    # Future risks reach 80 at step 3 -> 3 * 5 = 15 seconds
    future_risks = [30.0, 50.0, 80.0, 90.0, 95.0]
    time_sec, step_idx = ew_engine.estimate_time_to_high_risk(current_risk=20.0, future_risks=future_risks)
    assert time_sec == 15
    assert step_idx == 3

    # Already HIGH current risk -> 0 seconds
    time_sec_curr, _ = ew_engine.estimate_time_to_high_risk(current_risk=80.0, future_risks=future_risks)
    assert time_sec_curr == 0

    # No future high risk -> None
    time_sec_none, _ = ew_engine.estimate_time_to_high_risk(current_risk=20.0, future_risks=[30.0, 40.0, 50.0])
    assert time_sec_none is None

def test_rollout_dimensions():
    simulator = RolloutSimulator()
    dummy_seq = np.random.randn(2, 10, 69).astype(np.float32)
    res = simulator.rollout(dummy_seq, steps=5)

    assert res["future_states"].shape == (2, 5, 69)
    assert res["future_attack_probabilities"].shape == (2, 5, 1)
    assert res["future_stage_probabilities"].shape == (2, 5, 6)
    assert res["future_stage_predictions"].shape == (2, 5)

def test_inference_integration_smoke():
    """Runs full integration pipeline test using actual trained checkpoint and sequence data."""
    seq_file = PROJECT_ROOT / "data" / "processed" / "X_val_seq.npy"
    if not seq_file.exists():
        pytest.skip("X_val_seq.npy dataset file not found for integration test.")

    val_seqs = np.load(seq_file)
    test_seq = val_seqs[0:1] # First sample (1, 10, 69)

    ew_engine = EarlyWarningEngine()
    analysis = ew_engine.analyze_sequence(test_seq)

    assert "current_risk" in analysis
    assert "future_risks" in analysis
    assert len(analysis["future_risks"]) == 5
    assert "warning_message" in analysis

    print("\n" + "=" * 60)
    print(" INTEGRATION TEST REPORT: CyberWorld-AI Predictive Defence")
    print("=" * 60)
    print(f"Current Attack Probability : {analysis['current_attack_probability'] * 100:.2f}%")
    print(f"Current Risk Score         : {analysis['current_risk']:.2f}/100")
    print(f"Risk Level                 : {analysis['risk_level']}")
    print(f"Predicted Stage            : {analysis['predicted_stage']}")
    print("\nFuture Risk Forecast (5-Step Rollout):")
    for k, (r, p, stg) in enumerate(zip(analysis['future_risks'], analysis['future_attack_probabilities'], analysis['future_stages']), start=1):
        print(f"  +{(k*5):<2} sec : Risk {r:5.1f}/100 | Attack Prob: {p*100:5.1f}% | Stage: {stg}")
    print(f"\nTime to High Risk          : {analysis['time_to_high_risk']} seconds" if analysis['time_to_high_risk'] is not None else "\nTime to High Risk          : N/A (Low Risk)")
    print(f"Warning Advisory           : {analysis['warning_message']}")
    print("=" * 60 + "\n")
