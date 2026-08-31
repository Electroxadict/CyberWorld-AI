"""
XGBoost Predictor Engine for CyberWorld-AI.
Loads trained XGBoost risk and stage classifiers, extracts 489 temporal features via World Model rollout,
and provides unified prediction APIs with MITRE stage mapping explanation.
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

from preprocessing.check_dataset import load_config
from inference.predictor import WorldModelPredictor
from inference.rollout import RolloutSimulator
from training.train_xgboost import extract_temporal_features_from_sequence, build_xgb_feature_names
from attack_mapping.mitre_mapper import MitreStageMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class XGBoostPredictor:
    """Inference predictor for XGBoost Risk & Stage models using World Model rollout features."""
    
    def __init__(self, models_dir=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        if models_dir is None:
            self.models_dir = Path(self.config["paths"]["models_dir"])
        else:
            self.models_dir = Path(models_dir)
            
        self.risk_model = None
        self.stage_model = None
        self.xgb_feature_columns = []
        self.base_feature_columns = []
        self.model_config = {}
        self.is_loaded = False
        
        self.predictor = WorldModelPredictor(models_dir=self.models_dir, config_path=config_path)
        self.rollout_sim = RolloutSimulator(predictor=self.predictor, config_path=config_path)
        self.mapper = MitreStageMapper(config_path=config_path)
        
        self.load_models()

    def load_models(self):
        """Loads trained XGBoost model pickles, feature column lists, and config JSON."""
        risk_path = self.models_dir / "xgb_risk_model.pkl"
        stage_path = self.models_dir / "xgb_stage_model.pkl"
        xgb_cols_path = self.models_dir / "xgb_feature_columns.pkl"
        base_cols_path = self.models_dir / "feature_columns.pkl"
        config_path = self.models_dir / "xgb_model_config.json"
        
        if not risk_path.exists() or not stage_path.exists():
            raise FileNotFoundError(f"XGBoost model files not found in {self.models_dir}. Run train_xgboost.py first.")
            
        with open(risk_path, "rb") as f:
            self.risk_model = pickle.load(f)
            
        with open(stage_path, "rb") as f:
            self.stage_model = pickle.load(f)
            
        with open(xgb_cols_path, "rb") as f:
            self.xgb_feature_columns = pickle.load(f)
            
        with open(base_cols_path, "rb") as f:
            self.base_feature_columns = pickle.load(f)
            
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
                
        # Validate Feature Count
        expected_xgb_cols = build_xgb_feature_names(self.base_feature_columns)
        if len(self.xgb_feature_columns) != len(expected_xgb_cols):
            raise ValueError(f"XGBoost feature column count ({len(self.xgb_feature_columns)}) mismatch with expected ({len(expected_xgb_cols)}).")
            
        self.is_loaded = True
        logger.info(f"Loaded XGBoost Risk and Stage models successfully ({len(self.xgb_feature_columns)} features).")

    def build_features(self, x_seq) -> np.ndarray:
        """
        Extracts 489 temporal feature vectors from sequence input (N, 10, 69).
        """
        x_tensor = self.predictor.prepare_input(x_seq)
        x_np = x_tensor.cpu().numpy()
        features_2d = extract_temporal_features_from_sequence(x_np, self.rollout_sim, self.base_feature_columns)
        return features_2d

    def predict_attack_probability(self, x_seq) -> np.ndarray:
        """Predicts attack probability for sequence."""
        feat_2d = self.build_features(x_seq)
        probs = self.risk_model.predict_proba(feat_2d)[:, 1]
        return probs

    def predict_attack_stage(self, x_seq) -> tuple:
        """Predicts 6-class stage probabilities and stage prediction indices."""
        feat_2d = self.build_features(x_seq)
        raw_stage_probs = self.stage_model.predict_proba(feat_2d)
        batch_size = len(raw_stage_probs)
        
        full_stage_probs = np.zeros((batch_size, 6), dtype=np.float32)
        classes = getattr(self.stage_model, "classes_", range(raw_stage_probs.shape[1]))
        
        for idx, cls in enumerate(classes):
            if 0 <= cls < 6:
                full_stage_probs[:, cls] = raw_stage_probs[:, idx]
                
        stage_preds = np.argmax(full_stage_probs, axis=-1)
        return full_stage_probs, stage_preds

    def predict(self, x_seq) -> dict:
        """
        Runs combined XGBoost risk and stage inference with MITRE explanation.
        """
        feat_2d = self.build_features(x_seq)
        
        att_prob = float(self.risk_model.predict_proba(feat_2d)[0, 1])
        full_stage_probs, stage_preds = self.predict_attack_stage(x_seq)
        
        pred_stage_idx = int(stage_preds[0])
        stage_probs_list = full_stage_probs[0].tolist()
        
        stage_name = self.mapper.get_stage_name(pred_stage_idx)
        stage_desc = self.mapper.get_stage_description(pred_stage_idx)
        explanation = self.mapper.explain_mapping(pred_stage_idx)
        
        return {
            "attack_probability": att_prob,
            "predicted_stage": pred_stage_idx,
            "stage_name": stage_name,
            "stage_description": stage_desc,
            "stage_probabilities": stage_probs_list,
            "stage_reason": explanation["reason"],
            "model": "XGBoost"
        }

if __name__ == "__main__":
    xgb_pred = XGBoostPredictor()
    dummy_seq = np.random.randn(1, 10, 69).astype(np.float32)
    res = xgb_pred.predict(dummy_seq)
    print("XGBoost Predictor Test Passed:")
    print(f"  - Attack Prob : {res['attack_probability'] * 100:.2f}%")
    print(f"  - Stage Index : {res['predicted_stage']} ({res['stage_name']})")
    print(f"  - Stage Reason: {res['stage_reason']}")
