"""
K-Step Forward Simulation Engine for CyberWorld-AI.
Executes recursive multi-step temporal rollout using the World Model's own predicted states.
Forecasts future network state dynamics S[t+1] ... S[t+K] and threat trajectories without ground truth.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import torch

from preprocessing.check_dataset import load_config
from inference.predictor import WorldModelPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RolloutSimulator:
    """Multi-step forward simulator powered by Temporal World Model."""
    
    def __init__(self, predictor: WorldModelPredictor = None, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.prediction_horizon = self.config.get("temporal", {}).get("prediction_horizon", 5)
        
        if predictor is None:
            self.predictor = WorldModelPredictor(config_path=config_path)
        else:
            self.predictor = predictor

    def rollout(self, x_raw, steps: int = None) -> dict:
        """
        Executes K-step recursive forward simulation starting from sequence window x_raw.
        
        Args:
            x_raw (numpy.ndarray / torch.Tensor): Sequence window of shape (batch_size, sequence_length, num_features).
            steps (int, optional): Forecast horizon K. Defaults to config prediction_horizon (5).
            
        Returns:
            dict:
                - future_states: (batch_size, steps, num_features)
                - future_attack_probabilities: (batch_size, steps, 1)
                - future_stage_probabilities: (batch_size, steps, num_stages)
                - future_stage_predictions: (batch_size, steps)
        """
        if steps is None:
            steps = self.prediction_horizon
            
        x_tensor = self.predictor.prepare_input(x_raw)
        
        fut_states, fut_attack_probs, fut_stage_probs = self.predictor.model.rollout_future_states(
            x_tensor, steps=steps
        )
        
        fut_states_np = fut_states.cpu().numpy()
        fut_attack_probs_np = fut_attack_probs.cpu().numpy()
        fut_stage_probs_np = fut_stage_probs.cpu().numpy()
        fut_stage_preds_np = np.argmax(fut_stage_probs_np, axis=-1)
        
        return {
            "future_states": fut_states_np,
            "future_attack_probabilities": fut_attack_probs_np,
            "future_stage_probabilities": fut_stage_probs_np,
            "future_stage_predictions": fut_stage_preds_np
        }

if __name__ == "__main__":
    simulator = RolloutSimulator()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    res = simulator.rollout(dummy_seq, steps=5)
    print("Rollout Simulator Test Passed:")
    print(f"  - Future States Shape        : {res['future_states'].shape}")
    print(f"  - Future Attack Probs Shape  : {res['future_attack_probabilities'].shape}")
    print(f"  - Future Stage Probs Shape   : {res['future_stage_probabilities'].shape}")
    print(f"  - Future Stage Predictions   : {res['future_stage_predictions']}")
