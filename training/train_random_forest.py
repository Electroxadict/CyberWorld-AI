"""
Random Forest Classifier Trainer for CyberWorld-AI.
Trains Random Forest model on temporal network features, extracts feature importances
for dashboard explainability, evaluates metrics, and saves model artifacts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocessing.check_dataset import load_config
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.time_window import aggregate_time_windows
from training.train_logistic import evaluate_classifier_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_random_forest(data_path=None, config_path="config.yaml"):
    """
    Trains Random Forest Classifier with class balancing and feature importance logging.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load temporal aggregated windows
    time_windows_csv = Path(config["paths"]["processed_data"]).parent / "time_windows.csv"
    if not time_windows_csv.exists():
        logger.info("time_windows.csv not found. Running time_window aggregation...")
        df = aggregate_time_windows(config_path=config_path)
    else:
        df = pd.read_csv(time_windows_csv)
        
    logger.info(f"Loaded {len(df)} temporal state samples for Random Forest training.")
    
    reserved = ["Window_ID", "Window_Timestamp", "Attack", "Attack_Stage"]
    feature_cols = [c for c in df.columns if c not in reserved]
    
    X = df[feature_cols]
    y = df["Attack"].values
    
    # Chronological Split (60% Train, 20% Val, 20% Test)
    n = len(df)
    train_end = int(n * config["training"].get("train_split", 0.60))
    val_end = int(n * (config["training"].get("train_split", 0.60) + config["training"].get("val_split", 0.20)))
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    # Scale Features (Fitted ONLY on X_train)
    normalizer = FeatureNormalizer()
    X_train_scaled = normalizer.fit_transform(X_train, feature_cols=feature_cols)
    X_val_scaled = normalizer.transform(X_val)
    X_test_scaled = normalizer.transform(X_test)
    
    # Train Random Forest Classifier
    logger.info("Fitting RandomForestClassifier (n_estimators=100, class_weight='balanced')...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=config["training"].get("random_seed", 42))
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate on Test Set
    y_test_pred = rf.predict(X_test_scaled)
    y_test_prob = rf.predict_proba(X_test_scaled)[:, 1] if hasattr(rf, "predict_proba") else y_test_pred
    
    metrics = evaluate_classifier_metrics(y_test, y_test_pred, y_test_prob)
    
    print("\n" + "=" * 60)
    print(" RANDOM FOREST EVALUATION (TEST SET)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  - {k:<22}: {v:.4f}")
    print("=" * 60 + "\n")
    
    # Extract Feature Importances
    importances = rf.feature_importances_
    feat_imp_dict = {feat: float(imp) for feat, imp in zip(feature_cols, importances)}
    # Sort by importance descending
    sorted_feat_imp = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Save Model Artifacts
    model_path = models_dir / "random_forest.pkl"
    imp_path = models_dir / "feature_importance.json"
    
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
        
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(sorted_feat_imp, f, indent=2)
        
    normalizer.save(models_dir=models_dir, scaler_filename="rf_scaler.pkl", cols_filename="feature_columns.pkl")
    
    logger.info(f"Saved Random Forest model to {model_path}")
    logger.info(f"Saved feature importances to {imp_path}")
    
    return rf, metrics, sorted_feat_imp

if __name__ == "__main__":
    train_random_forest()
