"""
Model Artifact Integrity & Inference Validation Tool for CyberWorld-AI.
Loads all trained PyTorch, XGBoost, and StandardScaler artifacts, verifies feature dimensions,
and tests forward pass execution without retraining.

Usage: python scripts/validate_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import json
import torch
import numpy as np

from models.world_model import TemporalWorldModel

def validate_all_models():
    print("=" * 60)
    print("       CYBERWORLD-AI MODEL ARTIFACT VALIDATION")
    print("=" * 60)
    
    models_dir = PROJECT_ROOT / "models"
    all_ok = True
    
    # 1. Base Feature Schema & Scaler
    try:
        with open(models_dir / "feature_columns.pkl", "rb") as f:
            base_cols = pickle.load(f)
        with open(models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print(f"[PASS] Feature Schema & Scaler : {len(base_cols)} features loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Feature Schema / Scaler: {e}")
        all_ok = False
        
    # 2. PyTorch World Model
    try:
        wm_path = models_dir / "world_model.pt"
        cfg_path = models_dir / "model_config.json"
        with open(cfg_path, "r") as f:
            wm_cfg = json.load(f)
            
        model = TemporalWorldModel(num_features=len(base_cols))
        ckpt = torch.load(wm_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test forward pass
        dummy_x = torch.randn(1, 10, len(base_cols))
        with torch.no_grad():
            next_s, atk_logits, stg_logits, attn = model(dummy_x)
            
        assert next_s.shape == (1, len(base_cols))
        assert attn.shape == (1, 10)
        print(f"[PASS] LSTM Temporal World Model: Loaded state dict & verified forward pass (Hidden={wm_cfg.get('hidden_size', 128)}).")
    except Exception as e:
        print(f"[FAIL] LSTM Temporal World Model: {e}")
        all_ok = False
        
    # 3. XGBoost Risk & Stage Models
    try:
        with open(models_dir / "xgb_risk_model.pkl", "rb") as f:
            xgb_risk = pickle.load(f)
        with open(models_dir / "xgb_stage_model.pkl", "rb") as f:
            xgb_stage = pickle.load(f)
        with open(models_dir / "xgb_feature_columns.pkl", "rb") as f:
            xgb_cols = pickle.load(f)
            
        dummy_xgb_x = np.random.randn(2, len(xgb_cols)).astype(np.float32)
        risk_probs = xgb_risk.predict_proba(dummy_xgb_x)[:, 1]
        stage_probs = xgb_stage.predict_proba(dummy_xgb_x)
        
        assert len(xgb_cols) == 489
        assert len(risk_probs) == 2
        assert stage_probs.shape[1] > 0
        print(f"[PASS] XGBoost Risk & Stage    : Loaded 2 models ({len(xgb_cols)} features, {stage_probs.shape[1]} active stage classes).")
    except Exception as e:
        print(f"[FAIL] XGBoost Models         : {e}")
        all_ok = False
        
    print("=" * 60)
    if all_ok:
        print("OVERALL MODEL INTEGRITY: PASS (All model artifacts valid & runnable)")
    else:
        print("OVERALL MODEL INTEGRITY: FAIL (Issues detected in model artifacts)")
    print("=" * 60 + "\n")
    
    return all_ok

if __name__ == "__main__":
    ok = validate_all_models()
    sys.exit(0 if ok else 1)
