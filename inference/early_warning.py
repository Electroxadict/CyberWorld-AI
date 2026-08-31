"""
Early Warning Engine for CyberWorld-AI.
Analyzes multi-step risk forecasts, calculates time_to_high_risk in seconds,
decodes MITRE ATT&CK stage progression, and generates predictive threat warnings.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np

from preprocessing.check_dataset import load_config
from inference.predictor import WorldModelPredictor
from inference.rollout import RolloutSimulator
from inference.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EarlyWarningEngine:
    """Evaluates temporal threat forecasts and issues early warning alerts."""
    
    def __init__(
        self,
        predictor: WorldModelPredictor = None,
        rollout_sim: RolloutSimulator = None,
        risk_engine: RiskEngine = None,
        config_path="config.yaml"
    ):
        self.config = load_config(config_path)
        self.window_sec = self.config.get("temporal", {}).get("time_window_seconds", 5)
        self.horizon_steps = self.config.get("temporal", {}).get("prediction_horizon", 5)
        
        # Risk Thresholds
        risk_cfg = self.config.get("risk", {})
        self.high_threshold = risk_cfg.get("high", 75)
        self.critical_threshold = risk_cfg.get("critical", 90)
        
        # MITRE Stage Names Mapping
        self.stage_names = self.config.get("mitre_stages", {
            0: "Normal",
            1: "Reconnaissance",
            2: "Initial Access",
            3: "Lateral Movement",
            4: "Command and Control",
            5: "Exfiltration"
        })
        
        self.predictor = predictor if predictor else WorldModelPredictor(config_path=config_path)
        self.rollout_sim = rollout_sim if rollout_sim else RolloutSimulator(predictor=self.predictor, config_path=config_path)
        self.risk_engine = risk_engine if risk_engine else RiskEngine(config_path=config_path)

    def decode_stage_name(self, stage_idx: int) -> str:
        """Converts integer stage index (0..5) to human-readable MITRE stage name."""
        return self.stage_names.get(int(stage_idx), f"Stage-{stage_idx}")

    def estimate_time_to_high_risk(self, current_risk: float, future_risks: list) -> tuple:
        """
        Estimates approximate seconds until risk reaches HIGH threshold (75+).
        
        Returns:
            tuple: (time_to_high_risk_seconds, step_index)
                - time_to_high_risk_seconds: int or None if no high risk predicted.
                - step_index: 1-indexed step index or 0 if currently high.
        """
        if current_risk >= self.high_threshold:
            return 0, 0
            
        for k, risk_val in enumerate(future_risks, start=1):
            if risk_val >= self.high_threshold:
                approx_seconds = k * self.window_sec
                return approx_seconds, k
                
        return None, None

    def generate_warning_message(self, warning_triggered: bool, current_risk: float, time_to_high_risk: int, max_future_stage: str) -> str:
        """Constructs human-readable alert advisory message."""
        if not warning_triggered:
            return "System Status: NORMAL. Network activity is within safe operational baseline limits."
            
        if current_risk >= self.high_threshold:
            return (
                f"CRITICAL WARNING: Current network threat risk level is already CRITICAL ({current_risk:.1f}/100). "
                f"Active stage: {max_future_stage}. Immediate defensive investigation and host isolation advised."
            )
            
        if time_to_high_risk is not None:
            return (
                f"EARLY WARNING: High-risk threat progression predicted within approximately {time_to_high_risk} seconds. "
                f"Potential attack stage: {max_future_stage}. Proactive threat hunting recommended."
            )
            
        return f"ELEVATED RISK WARNING: Moderate threat activity detected (Risk Score: {current_risk:.1f}/100)."

    def analyze_sequence(self, x_raw) -> dict:
        """
        Runs full predictive pipeline: World Model -> 5-Step Rollout -> Risk Engine -> Early Warning.
        
        Args:
            x_raw (numpy.ndarray / torch.Tensor): Sequence window (batch_size=1 or tensor, seq_len=10, num_features).
            
        Returns:
            dict: Structured early warning analysis result.
        """
        # 1. Single Step Prediction
        pred_res = self.predictor.predict(x_raw)
        curr_att_prob = float(pred_res["attack_probability"][0, 0])
        curr_stage_idx = int(pred_res["stage_prediction"][0])
        curr_stage_name = self.decode_stage_name(curr_stage_idx)
        
        # 2. Multi-step Forward Simulation (Rollout)
        rollout_res = self.rollout_sim.rollout(x_raw, steps=self.horizon_steps)
        fut_states = rollout_res["future_states"][0]                     # (K, num_features)
        fut_att_probs = rollout_res["future_attack_probabilities"][0].ravel() # (K,)
        fut_stage_indices = rollout_res["future_stage_predictions"][0]   # (K,)
        fut_stage_names = [self.decode_stage_name(idx) for idx in fut_stage_indices]
        
        # 3. Calculate Current and Future Risks per step
        curr_state = x_raw[0, -1, :] if isinstance(x_raw, np.ndarray) else x_raw[0, -1, :].cpu().numpy()
        risk_res = self.risk_engine.calculate_risk(curr_att_prob, fut_att_probs, current_state=curr_state)
        curr_risk_score = risk_res["risk_score"]
        curr_risk_level = risk_res["risk_level"]
        
        future_risks = []
        for k in range(len(fut_att_probs)):
            step_att = fut_att_probs[k]
            step_fut = fut_att_probs[k:]
            step_risk = self.risk_engine.calculate_risk(step_att, step_fut, current_state=fut_states[k])["risk_score"]
            future_risks.append(float(step_risk))
            
        # 4. Early Warning Evaluation
        max_fut_risk = max([curr_risk_score] + future_risks)
        warning_triggered = bool(curr_risk_score >= self.high_threshold or max_fut_risk >= self.high_threshold)
        
        time_to_high_sec, step_idx = self.estimate_time_to_high_risk(curr_risk_score, future_risks)
        
        # Identify highest severity stage in forecast
        max_stage_idx = max([curr_stage_idx] + list(fut_stage_indices))
        max_stage_name = self.decode_stage_name(max_stage_idx)
        
        warning_msg = self.generate_warning_message(warning_triggered, curr_risk_score, time_to_high_sec, max_stage_name)
        
        return {
            "current_risk": curr_risk_score,
            "risk_level": curr_risk_level,
            "current_attack_probability": curr_att_prob,
            "predicted_stage": curr_stage_name,
            "predicted_stage_idx": curr_stage_idx,
            "future_risks": future_risks,
            "future_attack_probabilities": [float(p) for p in fut_att_probs],
            "future_stages": fut_stage_names,
            "warning_triggered": warning_triggered,
            "time_to_high_risk": time_to_high_sec,
            "warning_message": warning_msg,
            "risk_breakdown": risk_res
        }

if __name__ == "__main__":
    ew_engine = EarlyWarningEngine()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    res = ew_engine.analyze_sequence(dummy_seq)
    print("Early Warning Engine Test Passed:")
    print(f"  - Current Risk Score : {res['current_risk']:.2f}/100 ({res['risk_level']})")
    print(f"  - Future Risks       : {[round(r, 1) for r in res['future_risks']]}")
    print(f"  - Warning Triggered  : {res['warning_triggered']}")
    print(f"  - Time to High Risk  : {res['time_to_high_risk']}")
    print(f"  - Warning Message    : {res['warning_message']}")
