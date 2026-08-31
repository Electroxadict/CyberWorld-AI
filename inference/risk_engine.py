"""
Quantitative Risk Engine for CyberWorld-AI.
Calculates transparent 0-100 cyber threat risk scores by fusing current attack likelihood,
predicted multi-step progression probability, state anomaly metrics, and temporal risk trend.
Categorizes threat levels (LOW, MODERATE, HIGH, CRITICAL).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
import numpy as np

from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RiskEngine:
    """Computes transparent threat risk scores (0-100) and risk level categories."""
    
    def __init__(self, config_path="config.yaml", models_dir=None):
        self.config = load_config(config_path)
        risk_cfg = self.config.get("risk", {})
        
        # Component Weights (Sum = 1.0)
        self.w_current = risk_cfg.get("w_current", 0.60)
        self.w_progression = risk_cfg.get("w_progression", 0.20)
        self.w_anomaly = risk_cfg.get("w_anomaly", 0.10)
        self.w_trend = risk_cfg.get("w_trend", 0.10)
        
        # Risk Thresholds
        self.th_low = risk_cfg.get("low", 25)
        self.th_moderate = risk_cfg.get("moderate", 50)
        self.th_high = risk_cfg.get("high", 75)
        self.th_critical = risk_cfg.get("critical", 90)
        
        # Load Scaler for Reference Baseline (Mean & Variance)
        if models_dir is None:
            models_dir = Path(self.config["paths"]["models_dir"])
        else:
            models_dir = Path(models_dir)
            
        scaler_path = models_dir / "scaler.pkl"
        self.reference_mean = None
        self.reference_scale = None
        
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    self.reference_mean = scaler.mean_
                    self.reference_scale = scaler.scale_
            except Exception as e:
                logger.warning(f"Could not load reference statistics from {scaler_path}: {e}")

    def calculate_progression_probability(self, future_attack_probs: np.ndarray) -> float:
        """
        Calculates maximum attack probability across the K-step rollout horizon.
        
        Args:
            future_attack_probs (np.ndarray): Array of future attack probabilities [P[t+1]...P[t+K]].
            
        Returns:
            float: Maximum predicted future attack probability clamped in [0, 1].
        """
        probs_flat = np.array(future_attack_probs).ravel()
        if len(probs_flat) == 0:
            return 0.0
        max_p = float(np.max(probs_flat))
        return float(np.clip(max_p, 0.0, 1.0))

    def calculate_trend_score(self, current_attack_prob: float, future_attack_probs: np.ndarray) -> float:
        """
        Calculates normalized positive trend slope across predicted future states.
        Measures if attack likelihood is escalating over time.
        
        Args:
            current_attack_prob (float): Current time step attack probability P[t].
            future_attack_probs (np.ndarray): Future attack probabilities P[t+1]...P[t+K].
            
        Returns:
            float: Escalation trend score clamped in [0, 1].
        """
        probs = [float(current_attack_prob)] + [float(p) for p in np.array(future_attack_probs).ravel()]
        if len(probs) < 2:
            return 0.0
            
        # Positive differences (escalations)
        diffs = [max(0.0, probs[i] - probs[i-1]) for i in range(1, len(probs))]
        avg_increase = np.mean(diffs) if diffs else 0.0
        
        # Scale by remaining headroom above current probability
        headroom = 1.0 - float(current_attack_prob) + 1e-5
        trend_score = float(avg_increase / headroom)
        return float(np.clip(trend_score, 0.0, 1.0))

    def calculate_anomaly_score(self, current_state: np.ndarray) -> float:
        """
        Calculates standardized feature distance relative to baseline training mean & variance.
        
        Args:
            current_state (np.ndarray): Feature vector for current network state S[t].
            
        Returns:
            float: Anomaly score scaled to [0, 1] via hyperbolic tangent.
        """
        if current_state is None or self.reference_mean is None or self.reference_scale is None:
            return 0.0
            
        state_flat = np.array(current_state).ravel()
        if len(state_flat) != len(self.reference_mean):
            return 0.0
            
        # Standardized Z-score distance per feature
        z_scores = (state_flat - self.reference_mean) / (self.reference_scale + 1e-5)
        rms_z = np.sqrt(np.mean(z_scores ** 2))
        
        # Normalize RMS Z-score into [0, 1] range using tanh
        anomaly_score = float(np.tanh(rms_z / 3.0))
        return float(np.clip(anomaly_score, 0.0, 1.0))

    def calculate_risk(
        self,
        current_attack_prob: float,
        future_attack_probs: np.ndarray,
        current_state: np.ndarray = None
    ) -> dict:
        """
        Calculates overall risk score (0-100) and risk level.
        
        Returns:
            dict: Detailed risk breakdown including overall risk score and level.
        """
        p_current = float(np.clip(current_attack_prob, 0.0, 1.0))
        p_progression = self.calculate_progression_probability(future_attack_probs)
        s_trend = self.calculate_trend_score(p_current, future_attack_probs)
        s_anomaly = self.calculate_anomaly_score(current_state)
        
        raw_risk_score = 100.0 * (
            (self.w_current * p_current) +
            (self.w_progression * p_progression) +
            (self.w_anomaly * s_anomaly) +
            (self.w_trend * s_trend)
        )
        
        risk_score = float(np.clip(raw_risk_score, 0.0, 100.0))
        risk_level = self.classify_risk_level(risk_score)
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "current_attack_probability": p_current,
            "progression_probability": p_progression,
            "anomaly_score": s_anomaly,
            "trend_score": s_trend,
            "weights": {
                "current": self.w_current,
                "progression": self.w_progression,
                "anomaly": self.w_anomaly,
                "trend": self.w_trend
            }
        }

    def classify_risk_level(self, risk_score: float) -> str:
        """Categorizes 0-100 risk score into threat severity levels."""
        score = float(risk_score)
        if score >= self.th_critical:
            return "CRITICAL"
        elif score >= self.th_high:
            return "HIGH"
        elif score >= self.th_moderate:
            return "MODERATE"
        else:
            return "LOW"

if __name__ == "__main__":
    risk_engine = RiskEngine()
    cur_p = 0.45
    fut_p = np.array([[0.55], [0.68], [0.82], [0.88], [0.94]])
    res = risk_engine.calculate_risk(cur_p, fut_p)
    print("Risk Engine Test Passed:")
    print(f"  - Risk Score  : {res['risk_score']:.2f}/100")
    print(f"  - Risk Level  : {res['risk_level']}")
    print(f"  - Progression : {res['progression_probability']:.4f}")
    print(f"  - Trend Score : {res['trend_score']:.4f}")
