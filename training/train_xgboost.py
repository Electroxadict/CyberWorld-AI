"""
XGBoost Risk and Attack Stage Model Trainer for CyberWorld-AI.
Combines current network state vectors with PyTorch World Model 5-step rollout dynamics
to construct 489-dimension temporal features for gradient boosted attack and stage prediction.
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
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from preprocessing.check_dataset import load_config
from inference.predictor import WorldModelPredictor
from inference.rollout import RolloutSimulator
from training.train_logistic import evaluate_classifier_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_xgb_feature_names(base_feature_columns: list) -> list:
    """Constructs deterministic list of 489 feature column names."""
    feat_names = []
    
    # 1. Current State Features (69)
    for col in base_feature_columns:
        feat_names.append(f"curr_{col}")
        
    # 2. Future State Aggregations (69 * 5 = 345)
    for col in base_feature_columns:
        feat_names.append(f"fut_mean_{col}")
        feat_names.append(f"fut_max_{col}")
        feat_names.append(f"fut_min_{col}")
        feat_names.append(f"fut_diff_{col}")
        feat_names.append(f"fut_pct_{col}")
        feat_names.append(f"fut_slope_{col}")
        
    # 3. World Model Threat Trajectory Summary Stats (6)
    feat_names.extend([
        "wm_current_attack_prob",
        "wm_future_attack_mean",
        "wm_future_attack_max",
        "wm_future_attack_slope",
        "wm_future_stage_max_prob",
        "wm_future_stage_dominant"
    ])
    
    return feat_names

def extract_temporal_features_from_sequence(
    X_seq_batch: np.ndarray,
    rollout_sim: RolloutSimulator,
    base_feature_columns: list
) -> np.ndarray:
    """
    Extracts 489 temporal feature vectors from sequence batch (N, 10, 69) using World Model rollout.
    """
    batch_size, seq_len, num_base_features = X_seq_batch.shape
    
    # Execute 5-step World Model rollout simulation
    rollout_res = rollout_sim.rollout(X_seq_batch, steps=5)
    
    fut_states = rollout_res["future_states"]                     # (N, 5, 69)
    fut_attack_probs = rollout_res["future_attack_probabilities"] # (N, 5, 1)
    fut_stage_probs = rollout_res["future_stage_probabilities"]   # (N, 5, 6)
    
    # Current states S[t] (N, 69)
    curr_states = X_seq_batch[:, -1, :]
    
    # Current World Model attack probability P[t]
    curr_attack_probs = rollout_sim.predictor.predict_attack_probability(X_seq_batch) # (N, 1)
    
    feature_rows = []
    
    for i in range(batch_size):
        curr_s = curr_states[i]                       # (69,)
        fut_s = fut_states[i]                         # (5, 69)
        fut_a = fut_attack_probs[i].ravel()           # (5,)
        fut_stg = fut_stage_probs[i]                  # (5, 6)
        c_att = float(curr_attack_probs[i, 0])
        
        # Aggregated Future State Statistics
        f_mean = np.mean(fut_s, axis=0)               # (69,)
        f_max = np.max(fut_s, axis=0)                 # (69,)
        f_min = np.min(fut_s, axis=0)                 # (69,)
        f_diff = fut_s[-1] - curr_s                   # (69,)
        f_pct = (fut_s[-1] - curr_s) / (np.abs(curr_s) + 1e-5) # (69,)
        f_slope = (fut_s[-1] - fut_s[0]) / 4.0        # (69,)
        
        # World Model Trajectory Summaries
        fa_mean = float(np.mean(fut_a))
        fa_max = float(np.max(fut_a))
        fa_slope = float((fut_a[-1] - c_att) / 5.0)
        
        # Non-benign stage max probability across rollout
        stg_non_benign_max = float(np.max(fut_stg[:, 1:])) if fut_stg.shape[1] > 1 else 0.0
        stg_dominant = float(np.argmax(np.mean(fut_stg, axis=0)))
        
        # Interleave features matching build_xgb_feature_names order
        row = list(curr_s)
        for j in range(num_base_features):
            row.extend([
                f_mean[j],
                f_max[j],
                f_min[j],
                f_diff[j],
                f_pct[j],
                f_slope[j]
            ])
            
        row.extend([
            c_att,
            fa_mean,
            fa_max,
            fa_slope,
            stg_non_benign_max,
            stg_dominant
        ])
        
        feature_rows.append(row)
        
    return np.array(feature_rows, dtype=np.float32)

def train_xgboost_pipeline(config_path="config.yaml"):
    """
    Trains XGBoost Risk Model (binary) and XGBoost Stage Model (6-class) on World Model features.
    """
    config = load_config(config_path)
    seq_dir = Path(config["paths"]["sequences_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Sequence Datasets
    logger.info("Loading sequence dataset matrices...")
    X_train_seq = np.load(seq_dir / "X_train_seq.npy")
    Y_train_attack = np.load(seq_dir / "Y_train_attack.npy")
    Y_train_stage = np.load(seq_dir / "Y_train_stage.npy")
    
    X_val_seq = np.load(seq_dir / "X_val_seq.npy")
    Y_val_attack = np.load(seq_dir / "Y_val_attack.npy")
    Y_val_stage = np.load(seq_dir / "Y_val_stage.npy")
    
    X_test_seq = np.load(seq_dir / "X_test_seq.npy")
    Y_test_attack = np.load(seq_dir / "Y_test_attack.npy")
    Y_test_stage = np.load(seq_dir / "Y_test_stage.npy")
    
    # Load Base Feature Columns
    cols_path = models_dir / "feature_columns.pkl"
    with open(cols_path, "rb") as f:
        base_feature_columns = pickle.load(f)
        
    xgb_feat_names = build_xgb_feature_names(base_feature_columns)
    
    # 2. Instantiate World Model Rollout Simulator
    logger.info("Initializing World Model Predictor & Rollout Simulator for temporal feature construction...")
    predictor = WorldModelPredictor(models_dir=models_dir, config_path=config_path)
    rollout_sim = RolloutSimulator(predictor=predictor, config_path=config_path)
    
    # 3. Build Temporal Feature Matrices (NO GROUND TRUTH FUTURE STATES USED)
    logger.info(f"Building {len(xgb_feat_names)} temporal features for Training sequences ({len(X_train_seq)})...")
    X_train_xgb = extract_temporal_features_from_sequence(X_train_seq, rollout_sim, base_feature_columns)
    
    logger.info(f"Building temporal features for Validation sequences ({len(X_val_seq)})...")
    X_val_xgb = extract_temporal_features_from_sequence(X_val_seq, rollout_sim, base_feature_columns)
    
    logger.info(f"Building temporal features for Test sequences ({len(X_test_seq)})...")
    X_test_xgb = extract_temporal_features_from_sequence(X_test_seq, rollout_sim, base_feature_columns)
    
    # Read Hyperparameters from config.yaml
    xgb_cfg = config.get("xgboost", {})
    n_estimators = xgb_cfg.get("n_estimators", 200)
    max_depth = xgb_cfg.get("max_depth", 6)
    learning_rate = xgb_cfg.get("learning_rate", 0.05)
    subsample = xgb_cfg.get("subsample", 0.8)
    colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8)
    reg_lambda = xgb_cfg.get("reg_lambda", 1.0)
    random_state = xgb_cfg.get("random_seed", 42)
    
    # Calculate scale_pos_weight for binary risk model imbalance
    num_neg = np.sum(Y_train_attack == 0)
    num_pos = np.sum(Y_train_attack == 1)
    scale_pos_weight = float(num_neg / (num_pos + 1e-5))
    
    # --- 4. Train XGBoost Risk Model (Binary Attack Detection) ---
    logger.info("Training XGBoost Risk Model (Binary Attack Classifier)...")
    xgb_risk_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss"
    )
    xgb_risk_model.fit(
        X_train_xgb, Y_train_attack,
        eval_set=[(X_val_xgb, Y_val_attack)],
        verbose=False
    )
    
    # Evaluate Risk Model on Chronological Test Set
    y_test_risk_pred = xgb_risk_model.predict(X_test_xgb)
    y_test_risk_prob = xgb_risk_model.predict_proba(X_test_xgb)[:, 1]
    
    risk_metrics = evaluate_classifier_metrics(Y_test_attack, y_test_risk_pred, y_test_risk_prob)
    
    print("\n" + "=" * 60)
    print(" XGBOOST RISK MODEL EVALUATION (CHRONOLOGICAL TEST SET)")
    print("=" * 60)
    print(f"Train samples: {len(X_train_xgb)} | Val samples: {len(X_val_xgb)} | Test samples: {len(X_test_xgb)}")
    for k, v in risk_metrics.items():
        print(f"  - {k:<22}: {v:.4f}")
    print("=" * 60 + "\n")
    
    # Save Risk Model Artifacts
    risk_model_path = models_dir / "xgb_risk_model.pkl"
    with open(risk_model_path, "wb") as f:
        pickle.dump(xgb_risk_model, f)
        
    xgb_cols_path = models_dir / "xgb_feature_columns.pkl"
    with open(xgb_cols_path, "wb") as f:
        pickle.dump(xgb_feat_names, f)
        
    xgb_config_json = {
        "num_base_features": len(base_feature_columns),
        "num_xgb_features": len(xgb_feat_names),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "feature_columns": xgb_feat_names
    }
    with open(models_dir / "xgb_model_config.json", "w", encoding="utf-8") as f:
        json.dump(xgb_config_json, f, indent=2)
        
    logger.info(f"Saved XGBoost Risk Model to {risk_model_path}")
    logger.info(f"Saved XGBoost feature list to {xgb_cols_path}")
    
    # --- 5. Train XGBoost Attack Stage Model (6-Class Multi-Softprob) ---
    logger.info("Training XGBoost Attack Stage Model (6-Class Multi-Softprob)...")
    xgb_stage_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="mlogloss"
    )
    xgb_stage_model.fit(
        X_train_xgb, Y_train_stage,
        eval_set=[(X_val_xgb, Y_val_stage)],
        verbose=False
    )
    
    # Evaluate Stage Model on Test Set
    y_test_stage_pred = xgb_stage_model.predict(X_test_xgb)
    stage_acc = accuracy_score(Y_test_stage, y_test_stage_pred)
    stage_f1_macro = f1_score(Y_test_stage, y_test_stage_pred, average="macro", zero_division=0)
    stage_prec_macro = precision_score(Y_test_stage, y_test_stage_pred, average="macro", zero_division=0)
    stage_rec_macro = recall_score(Y_test_stage, y_test_stage_pred, average="macro", zero_division=0)
    
    print("=" * 60)
    print(" XGBOOST ATTACK STAGE MODEL EVALUATION (CHRONOLOGICAL TEST SET)")
    print("=" * 60)
    print(f"  - Stage Accuracy    : {stage_acc:.4f}")
    print(f"  - Stage Macro F1    : {stage_f1_macro:.4f}")
    print(f"  - Stage Macro Prec  : {stage_prec_macro:.4f}")
    print(f"  - Stage Macro Rec   : {stage_rec_macro:.4f}")
    print("=" * 60 + "\n")
    
    # Save Stage Model Artifact
    stage_model_path = models_dir / "xgb_stage_model.pkl"
    with open(stage_model_path, "wb") as f:
        pickle.dump(xgb_stage_model, f)
    logger.info(f"Saved XGBoost Stage Model to {stage_model_path}")
    
    # Save Stage Confusion Matrix Plot
    cm = confusion_matrix(Y_test_stage, y_test_stage_pred, labels=list(range(6)))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Recon", "Access", "Lateral", "C2", "Exfil"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("XGBoost Attack Stage Confusion Matrix")
    plt.tight_layout()
    cm_path = logs_dir / "xgb_stage_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")
    
    # Extract Feature Importances (Top 20)
    importances = xgb_risk_model.feature_importances_
    feat_imp_dict = {name: float(imp) for name, imp in zip(xgb_feat_names, importances)}
    sorted_feat_imp = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))
    
    imp_json_path = logs_dir / "xgb_feature_importance.json"
    with open(imp_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_feat_imp, f, indent=2)
    logger.info(f"Saved XGBoost feature importances to {imp_json_path}")
    
    # Plot Top 20 Feature Importances
    top_20 = list(sorted_feat_imp.items())[:20]
    top_names = [item[0] for item in top_20][::-1]
    top_vals = [item[1] for item in top_20][::-1]
    
    plt.figure(figsize=(10, 7))
    plt.barh(top_names, top_vals, color="navy")
    plt.xlabel("XGBoost Feature Importance")
    plt.title("Top 20 Temporal & Network Features (XGBoost)")
    plt.tight_layout()
    imp_img_path = logs_dir / "xgb_feature_importance.png"
    plt.savefig(imp_img_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance plot to {imp_img_path}")
    
    print("\n" + "!" * 80)
    print(" WARNING: These are synthetic smoke-test results and must NOT be used as real CIC-IDS2018 benchmark results.")
    print("!" * 80 + "\n")
    
    return xgb_risk_model, xgb_stage_model, risk_metrics, {
        "accuracy": stage_acc,
        "f1_macro": stage_f1_macro,
        "precision_macro": stage_prec_macro,
        "recall_macro": stage_rec_macro
    }

if __name__ == "__main__":
    train_xgboost_pipeline()
