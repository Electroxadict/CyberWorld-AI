"""
Temporal Attention Visualizer and Explainer for CyberWorld-AI.
Extracts PyTorch World Model temporal window attention weights (a[t-9] ... a[t]),
identifies most influential historical network state time windows, plots attention distribution,
and provides combined SHAP + Attention threat explanation summaries for dashboard consumption.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

from preprocessing.check_dataset import load_config
from inference.predictor import WorldModelPredictor
from inference.xgboost_predictor import XGBoostPredictor
from explainability.shap_explainer import SHAPExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Timestep Labels for 10-step sequence window (S[t-9] ... S[t])
TIMESTEP_LABELS = ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"]

class TemporalAttentionVisualizer:
    """Extracts and visualizes PyTorch World Model temporal attention weights across historical time windows."""
    
    def __init__(self, predictor: WorldModelPredictor = None, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.logs_dir = Path(self.config["paths"]["logs_dir"])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if predictor is None:
            self.predictor = WorldModelPredictor(config_path=config_path)
        else:
            self.predictor = predictor
            
        self.window_sec = self.config.get("temporal", {}).get("time_window_seconds", 5)

    def get_temporal_attention(self, sequence) -> dict:
        """
        Extracts temporal attention weight vector for an input sequence.
        
        Args:
            sequence (numpy.ndarray / torch.Tensor): Sequence window of shape (1, 10, 69) or (10, 69).
            
        Returns:
            dict: {timesteps: list of str, attention_weights: list of float}
        """
        x_tensor = self.predictor.prepare_input(sequence)
        
        with torch.no_grad():
            _, _, _, attn_weights_tensor = self.predictor.model(x_tensor)
            
        weights = attn_weights_tensor[0].cpu().numpy().tolist()
        
        # Verify length == 10 and normalization sum ~ 1.0
        assert len(weights) == len(TIMESTEP_LABELS), f"Attention weights count ({len(weights)}) mismatch with sequence length ({len(TIMESTEP_LABELS)})"
        assert abs(sum(weights) - 1.0) < 0.05, f"Attention weights sum ({sum(weights):.4f}) deviates from Softmax normalization (1.0)."
        
        return {
            "timesteps": TIMESTEP_LABELS,
            "attention_weights": [float(w) for w in weights]
        }

    def get_most_influential_timestep(self, sequence) -> tuple:
        """
        Identifies historical timestep index and label with highest attention weight.
        
        Returns:
            tuple: (timestep_label, timestep_index, max_weight)
        """
        attn_dict = self.get_temporal_attention(sequence)
        weights = attn_dict["attention_weights"]
        
        max_idx = int(np.argmax(weights))
        max_label = TIMESTEP_LABELS[max_idx]
        max_weight = float(weights[max_idx])
        
        return max_label, max_idx, max_weight

    def generate_temporal_explanation(self, sequence) -> str:
        """
        Generates dynamic text explaining which historical time window influenced the World Model most.
        """
        max_label, max_idx, max_weight = self.get_most_influential_timestep(sequence)
        
        # Calculate approximate seconds ago
        steps_ago = (len(TIMESTEP_LABELS) - 1) - max_idx
        seconds_ago = steps_ago * self.window_sec
        
        weight_pct = max_weight * 100.0
        
        if steps_ago == 0:
            return f"The World Model placed the highest temporal attention ({weight_pct:.1f}%) on the current window ('t'), indicating immediate traffic activity has the strongest impact."
        else:
            return (
                f"The World Model placed the highest temporal attention ({weight_pct:.1f}%) on window '{max_label}', "
                f"indicating network behavior approximately {seconds_ago} seconds earlier had the strongest influence on predicted future states."
            )

    def plot_temporal_attention(self, sequence, output_path=None) -> Path:
        """
        Generates bar chart visualization of temporal attention weights across past timesteps t-9 ... t.
        """
        if output_path is None:
            output_path = self.logs_dir / "temporal_attention.png"
        else:
            output_path = Path(output_path)
            
        attn_dict = self.get_temporal_attention(sequence)
        labels = attn_dict["timesteps"]
        weights = attn_dict["attention_weights"]
        
        max_idx = int(np.argmax(weights))
        
        plt.figure(figsize=(9, 5))
        bars = plt.bar(labels, weights)
        
        # Highlight most influential timestep bar
        bars[max_idx].set_edgecolor("black")
        bars[max_idx].set_linewidth(1.5)
        
        plt.xlabel("Historical Network State Time Windows")
        plt.ylabel("Temporal Attention Weight (Softmax Normalized)")
        plt.title("LSTM World Model Temporal Attention Distribution (t-9 ... t)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved temporal attention plot to {output_path}")
        return output_path

class CyberWorldExplainer:
    """Unified Explainability interface combining XGBoost SHAP and World Model Temporal Attention."""
    
    def __init__(self, models_dir=None, config_path="config.yaml"):
        self.shap_explainer = SHAPExplainer(models_dir=models_dir, config_path=config_path)
        self.attn_visualizer = TemporalAttentionVisualizer(config_path=config_path)
        self.xgb_predictor = XGBoostPredictor(models_dir=models_dir, config_path=config_path)

    def explain_full_prediction(self, x_seq: np.ndarray) -> dict:
        """
        Executes unified explainability pipeline for input sequence window S[t-9] ... S[t].
        
        Returns:
            dict: Combined dictionary containing SHAP feature contributions, group shares, and temporal attention.
        """
        # 1. XGBoost Risk & Stage Inference
        xgb_res = self.xgb_predictor.predict(x_seq)
        
        # 2. Extract XGBoost 489-dimension temporal features for SHAP
        xgb_feat_2d = self.xgb_predictor.build_features(x_seq)
        
        # 3. Compute Local SHAP Explanation
        local_shap = self.shap_explainer.explain_prediction(xgb_feat_2d[0])
        group_shap = self.shap_explainer.group_shap_features(xgb_feat_2d)
        
        risk_explanation = self.shap_explainer.generate_explanation(
            xgb_feat_2d[0],
            risk_level="HIGH" if xgb_res["attack_probability"] > 0.6 else "SAFE",
            attack_prob=xgb_res["attack_probability"]
        )
        
        # 4. Compute Temporal Attention Explanation
        attn_dict = self.attn_visualizer.get_temporal_attention(x_seq)
        max_label, max_idx, max_weight = self.attn_visualizer.get_most_influential_timestep(x_seq)
        temp_explanation = self.attn_visualizer.generate_temporal_explanation(x_seq)
        
        # Generate and save plots
        self.attn_visualizer.plot_temporal_attention(x_seq)
        
        return {
            "attack_probability": xgb_res["attack_probability"],
            "predicted_stage": xgb_res["stage_name"],
            "risk_explanation": risk_explanation,
            "top_shap_features": local_shap[:10],
            "feature_group_importance": group_shap,
            "attention_timesteps": attn_dict["timesteps"],
            "attention_weights": attn_dict["attention_weights"],
            "most_influential_timestep": max_label,
            "most_influential_weight": max_weight,
            "temporal_explanation": temp_explanation
        }

if __name__ == "__main__":
    attn_vis = TemporalAttentionVisualizer()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    attn_res = attn_vis.get_temporal_attention(dummy_seq)
    temp_exp = attn_vis.generate_temporal_explanation(dummy_seq)
    attn_vis.plot_temporal_attention(dummy_seq)
    
    print("Temporal Attention Visualizer Test Passed:")
    print(f"  - Timesteps Count    : {len(attn_res['timesteps'])}")
    print(f"  - Weights Sum       : {sum(attn_res['attention_weights']):.4f}")
    print("\nTemporal Explanation Text:\n" + temp_exp)
