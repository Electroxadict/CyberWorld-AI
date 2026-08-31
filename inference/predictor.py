"""
Inference Predictor Engine for CyberWorld-AI.
Loads trained PyTorch World Model checkpoints, config parameters, and scalers.
Provides single-step prediction and state validation without retraining.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from preprocessing.check_dataset import load_config
from models.world_model import TemporalWorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WorldModelPredictor:
    """Inference interface wrapper for trained Temporal World Model."""
    
    def __init__(self, models_dir=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        if models_dir is None:
            self.models_dir = Path(self.config["paths"]["models_dir"])
        else:
            self.models_dir = Path(models_dir)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_config = {}
        self.feature_columns = []
        self.scaler = None
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        """Loads model weights, configuration, and feature metadata."""
        checkpoint_path = self.models_dir / "world_model.pt"
        config_json_path = self.models_dir / "model_config.json"
        scaler_path = self.models_dir / "scaler.pkl"
        cols_path = self.models_dir / "feature_columns.pkl"
        
        if not checkpoint_path.exists() or not config_json_path.exists():
            raise FileNotFoundError(f"Model checkpoint or config not found in {self.models_dir}. Run train_world_model.py first.")
            
        # Load configuration JSON
        with open(config_json_path, "r", encoding="utf-8") as f:
            self.model_config = json.load(f)
            
        num_features = self.model_config["num_features"]
        hidden_size = self.model_config.get("hidden_size", 128)
        num_layers = self.model_config.get("num_layers", 2)
        dropout = self.model_config.get("dropout", 0.2)
        num_stages = self.model_config.get("num_stages", 6)
        
        # Instantiate Model Architecture
        self.model = TemporalWorldModel(
            num_features=num_features,
            embedding_size=self.model_config.get("embedding_size", 64),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_stages=num_stages
        ).to(self.device)
        
        # Load Checkpoint Weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Load Feature Columns & Scaler
        if cols_path.exists():
            with open(cols_path, "rb") as f:
                self.feature_columns = pickle.load(f)
        else:
            self.feature_columns = self.model_config.get("feature_columns", [])
            
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                
        # Validate Feature Count
        if len(self.feature_columns) != num_features:
            raise ValueError(f"Feature columns count ({len(self.feature_columns)}) mismatch with model num_features ({num_features}).")
            
        self.is_loaded = True
        logger.info(f"Successfully loaded TemporalWorldModel onto device '{self.device.type.upper()}' with {num_features} features.")

    def prepare_input(self, x_raw) -> torch.Tensor:
        """
        Validates and converts input tensor/ndarray/DataFrame into PyTorch float tensor.
        Expected input shape: (batch_size, sequence_length, num_features)
        """
        if isinstance(x_raw, np.ndarray):
            x_tensor = torch.tensor(x_raw, dtype=torch.float32)
        elif isinstance(x_raw, torch.Tensor):
            x_tensor = x_raw.float()
        else:
            raise TypeError(f"Unsupported input type {type(x_raw)}. Expected numpy ndarray or torch Tensor.")
            
        if x_tensor.ndim == 2:
            # Single sequence sample (sequence_length, num_features) -> add batch dim
            x_tensor = x_tensor.unsqueeze(0)
            
        if x_tensor.ndim != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (batch, seq_len, num_features). Got shape {x_tensor.shape}")
            
        expected_seq_len = self.model_config.get("sequence_length", 10)
        expected_features = self.model_config["num_features"]
        
        if x_tensor.shape[1] != expected_seq_len:
            raise ValueError(f"Input sequence length ({x_tensor.shape[1]}) does not match expected length ({expected_seq_len}).")
            
        if x_tensor.shape[2] != expected_features:
            raise ValueError(f"Input feature count ({x_tensor.shape[2]}) does not match model num_features ({expected_features}).")
            
        return x_tensor.to(self.device)

    def predict(self, x_raw) -> dict:
        """
        Runs inference pass and returns structured dictionary of predictions.
        """
        x_tensor = self.prepare_input(x_raw)
        with torch.no_grad():
            pred_state, attack_logits, stage_logits, attn_weights = self.model(x_tensor)
            attack_prob = torch.sigmoid(attack_logits).cpu().numpy()
            stage_prob = F.softmax(stage_logits, dim=-1).cpu().numpy()
            stage_pred = np.argmax(stage_prob, axis=-1)
            
        return {
            "predicted_next_state": pred_state.cpu().numpy(),
            "attack_probability": attack_prob,
            "stage_probabilities": stage_prob,
            "stage_prediction": stage_pred,
            "attention_weights": attn_weights.cpu().numpy()
        }

    def predict_next_state(self, x_raw) -> np.ndarray:
        """Predicts next network state S[t+1]."""
        x_tensor = self.prepare_input(x_raw)
        pred_state = self.model.predict_next_state(x_tensor)
        return pred_state.cpu().numpy()

    def predict_attack_probability(self, x_raw) -> np.ndarray:
        """Predicts attack probability (0.0 to 1.0)."""
        x_tensor = self.prepare_input(x_raw)
        attack_prob = self.model.predict_attack_probability(x_tensor)
        return attack_prob.cpu().numpy()

    def predict_attack_stage(self, x_raw) -> np.ndarray:
        """Predicts attack stage probabilities across 6 MITRE classes."""
        x_tensor = self.prepare_input(x_raw)
        stage_prob = self.model.predict_attack_stage(x_tensor)
        return stage_prob.cpu().numpy()

if __name__ == "__main__":
    predictor = WorldModelPredictor()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    res = predictor.predict(dummy_seq)
    print("Predictor Test Passed:")
    print(f"  - Next State Shape : {res['predicted_next_state'].shape}")
    print(f"  - Attack Prob      : {res['attack_probability'].item():.4f}")
    print(f"  - Stage Pred       : {res['stage_prediction'].item()}")
