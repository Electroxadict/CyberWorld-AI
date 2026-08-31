"""
Baseline Logistic Regression Trainer for CyberWorld-AI.
Trains a linear baseline classifier on normalized temporal network features,
uses chronological splitting to prevent data leakage, and saves model artifacts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from preprocessing.check_dataset import load_config
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.time_window import aggregate_time_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_classifier_metrics(y_true, y_pred, y_prob=None):
    """Computes Accuracy, Precision, Recall, F1, FPR, and ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp) / (float(fp + tn) + 1e-10)
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else 0.5
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "False Positive Rate": fpr,
        "ROC-AUC": auc
    }

def train_logistic_baseline(data_path=None, config_path="config.yaml"):
    """
    Trains Logistic Regression baseline on chronological train/val/test splits.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate temporal aggregated windows
    time_windows_csv = Path(config["paths"]["processed_data"]).parent / "time_windows.csv"
    if not time_windows_csv.exists():
        logger.info("time_windows.csv not found. Running time_window aggregation...")
        df = aggregate_time_windows(config_path=config_path)
    else:
        df = pd.read_csv(time_windows_csv)
        
    logger.info(f"Loaded {len(df)} temporal state samples for baseline training.")
    
    # Define features and targets
    reserved = ["Window_ID", "Window_Timestamp", "Attack", "Attack_Stage"]
    feature_cols = [c for c in df.columns if c not in reserved]
    
    X = df[feature_cols]
    y = df["Attack"].values
    
    # --- Chronological Data Splitting (60% Train, 20% Val, 20% Test) ---
    n = len(df)
    train_end = int(n * config["training"].get("train_split", 0.60))
    val_end = int(n * (config["training"].get("train_split", 0.60) + config["training"].get("val_split", 0.20)))
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    logger.info(f"Chronological split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Fit normalizer ONLY on X_train to prevent target/data leakage
    normalizer = FeatureNormalizer()
    X_train_scaled = normalizer.fit_transform(X_train, feature_cols=feature_cols)
    X_val_scaled = normalizer.transform(X_val)
    X_test_scaled = normalizer.transform(X_test)
    
    # Train Logistic Regression Model
    logger.info("Fitting Logistic Regression model with class_weight='balanced'...")
    clf = LogisticRegression(class_weight="balanced", random_state=config["training"].get("random_seed", 42), max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on Test Set
    y_test_pred = clf.predict(X_test_scaled)
    y_test_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_test_pred
    
    metrics = evaluate_classifier_metrics(y_test, y_test_pred, y_test_prob)
    
    print("\n" + "=" * 60)
    print(" LOGISTIC REGRESSION BASELINE EVALUATION (TEST SET)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  - {k:<22}: {v:.4f}")
    print("=" * 60 + "\n")
    
    # Save model and scaler
    model_path = models_dir / "logistic_model.pkl"
    scaler_path = models_dir / "logistic_scaler.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    normalizer.save(models_dir=models_dir, scaler_filename="logistic_scaler.pkl", cols_filename="feature_columns.pkl")
    logger.info(f"Saved Logistic Regression model to {model_path}")
    logger.info(f"Saved Logistic Scaler to {scaler_path}")
    
    return clf, metrics

if __name__ == "__main__":
    train_logistic_baseline()
