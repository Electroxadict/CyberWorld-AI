"""
SHAP Feature Explainer for CyberWorld-AI.
Calculates global and local TreeSHAP explanations for trained XGBoost Risk Models,
aggregates SHAP values into macro feature categories, generates visualization plots,
and provides dynamic human-readable threat explanations. Includes compatibility patches for XGBoost 3.x.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import builtins
import logging
import json
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import xgboost
import shap.explainers._tree as st_module

from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- SHAP 0.49.x / XGBoost 3.x Compatibility Patch ---
# Strips bracketed base_score strings like '[4.9999994E-1]' during XGBTreeModelLoader initialization
_orig_float = builtins.float
_orig_loader_init = st_module.XGBTreeModelLoader.__init__

def _safe_loader_init(self, xgb_model):
    def _safe_float(val):
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            val = val.strip('[]')
        return _orig_float(val)
    builtins.float = _safe_float
    try:
        _orig_loader_init(self, xgb_model)
    finally:
        builtins.float = _orig_float

st_module.XGBTreeModelLoader.__init__ = _safe_loader_init
# ----------------------------------------------------

FEATURE_GROUP_PREFIXES = {
    "curr_": "CURRENT",
    "fut_mean_": "FUTURE MEAN",
    "fut_max_": "FUTURE MAX",
    "fut_min_": "FUTURE MIN",
    "fut_diff_": "FUTURE DIFFERENCE",
    "fut_pct_": "FUTURE PERCENTAGE CHANGE",
    "fut_slope_": "FUTURE SLOPE",
    "wm_": "WORLD MODEL THREAT FEATURES"
}

class SHAPExplainer:
    """Manages SHAP explanations for XGBoost Risk Model predictions."""
    
    def __init__(self, models_dir=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        if models_dir is None:
            self.models_dir = Path(self.config["paths"]["models_dir"])
        else:
            self.models_dir = Path(models_dir)
            
        self.logs_dir = Path(self.config["paths"]["logs_dir"])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.top_n = self.config.get("explainability", {}).get("top_features", 20)
        self.sample_size = self.config.get("explainability", {}).get("shap_sample_size", 200)
        
        self.risk_model = None
        self.xgb_feature_columns = []
        self.explainer = None
        
        self.load_model_and_explainer()

    def load_model_and_explainer(self):
        """Loads trained XGBoost model and initializes shap.TreeExplainer."""
        risk_path = self.models_dir / "xgb_risk_model.pkl"
        cols_path = self.models_dir / "xgb_feature_columns.pkl"
        
        if not risk_path.exists() or not cols_path.exists():
            raise FileNotFoundError(f"XGBoost model artifacts not found in {self.models_dir}. Run train_xgboost.py first.")
            
        with open(risk_path, "rb") as f:
            self.risk_model = pickle.load(f)
            
        with open(cols_path, "rb") as f:
            self.xgb_feature_columns = pickle.load(f)
            
        logger.info(f"Initializing shap.TreeExplainer (SHAP v{shap.__version__}, XGBoost v{xgboost.__version__})...")
        start_t = time.time()
        try:
            self.explainer = shap.TreeExplainer(self.risk_model)
        except Exception as e:
            logger.error(f"Failed to initialize shap.TreeExplainer: {e}")
            raise RuntimeError(f"SHAP TreeExplainer incompatibility: {e}")
            
        elapsed = time.time() - start_t
        logger.info(f"SHAP TreeExplainer initialized successfully in {elapsed:.3f} seconds for {len(self.xgb_feature_columns)} features.")

    def compute_global_shap(self, X_matrix: np.ndarray, sample_size: int = None) -> list:
        """
        Computes global mean absolute SHAP feature importance over a representative dataset sample.
        
        Returns:
            list of dicts: Sorted list of features with mean absolute SHAP values.
        """
        if sample_size is None:
            sample_size = self.sample_size
            
        N = len(X_matrix)
        if N > sample_size:
            indices = np.random.choice(N, sample_size, replace=False)
            X_sample = X_matrix[indices]
        else:
            X_sample = X_matrix
            
        logger.info(f"Computing global SHAP values across {len(X_sample)} background samples...")
        start_t = time.time()
        
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # For binary classifier list outputs
            
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        elapsed = time.time() - start_t
        logger.info(f"Global SHAP calculation finished in {elapsed:.3f}s.")
        
        # Sort features descending
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        global_importance = []
        for idx in sorted_indices:
            feat_name = self.xgb_feature_columns[idx] if idx < len(self.xgb_feature_columns) else f"feature_{idx}"
            global_importance.append({
                "feature": feat_name,
                "mean_abs_shap": float(mean_abs_shap[idx])
            })
            
        # Save JSON artifact
        json_path = self.logs_dir / "shap_global_importance.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(global_importance, f, indent=2)
        logger.info(f"Saved global SHAP importance to {json_path}")
        
        # Save PNG Plot (Top 20)
        self.plot_global_importance(global_importance[:self.top_n], self.logs_dir / "shap_global_importance.png")
        
        # Compute & save feature group importance
        self.group_shap_features(shap_values)
        
        return global_importance

    def plot_global_importance(self, top_features: list, output_path: Path):
        """Generates matplotlib bar chart of top global SHAP feature importances."""
        names = [f["feature"] for f in top_features][::-1]
        vals = [f["mean_abs_shap"] for f in top_features][::-1]
        
        plt.figure(figsize=(10, 7))
        plt.barh(names, vals)
        plt.xlabel("Mean Absolute SHAP Value (Global Impact)")
        plt.title(f"Top {len(names)} Global Feature Importances (SHAP TreeExplainer)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved global SHAP plot to {output_path}")

    def explain_prediction(self, x_vec: np.ndarray) -> list:
        """
        Calculates local SHAP explanation for a single prediction sample.
        
        Args:
            x_vec (np.ndarray): Single feature vector of shape (1, 489) or (489,).
            
        Returns:
            list of dicts: Feature contributions with direction and importance.
        """
        if x_vec.ndim == 1:
            x_sample = x_vec.reshape(1, -1)
        else:
            x_sample = x_vec
            
        shap_vals = self.explainer.shap_values(x_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
            
        s_vals = shap_vals[0]
        x_vals = x_sample[0]
        
        contributions = []
        for idx, (s_val, x_val) in enumerate(zip(s_vals, x_vals)):
            feat_name = self.xgb_feature_columns[idx] if idx < len(self.xgb_feature_columns) else f"feature_{idx}"
            direction = "INCREASES_RISK" if s_val > 0 else ("DECREASES_RISK" if s_val < 0 else "NEUTRAL")
            contributions.append({
                "feature": feat_name,
                "value": float(x_val),
                "shap_value": float(s_val),
                "direction": direction,
                "importance": float(abs(s_val))
            })
            
        # Sort descending by absolute SHAP importance
        contributions.sort(key=lambda item: item["importance"], reverse=True)
        
        # Save local explanation plot
        self.plot_local_explanation(contributions[:10], self.logs_dir / "shap_local_explanation.png")
        
        return contributions

    def plot_local_explanation(self, local_top_features: list, output_path: Path):
        """Generates matplotlib waterfall/bar plot of local SHAP feature contributions."""
        names = [f["feature"] for f in local_top_features][::-1]
        s_vals = [f["shap_value"] for f in local_top_features][::-1]
        
        colors = ["red" if v > 0 else "blue" for v in s_vals]
        
        plt.figure(figsize=(10, 6))
        plt.barh(names, s_vals, color=colors)
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("SHAP Contribution (+ Increases Risk / - Decreases Risk)")
        plt.title("Local Feature Contribution (SHAP Explanation)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved local SHAP plot to {output_path}")

    def group_shap_features(self, shap_matrix: np.ndarray) -> dict:
        """
        Groups the 489 features into 8 logical macro categories and computes aggregate importance.
        """
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        
        group_sums = {group_name: 0.0 for group_name in FEATURE_GROUP_PREFIXES.values()}
        group_counts = {group_name: 0 for group_name in FEATURE_GROUP_PREFIXES.values()}
        
        for idx, feat_name in enumerate(self.xgb_feature_columns):
            val = float(mean_abs[idx])
            matched = False
            for prefix, group_name in FEATURE_GROUP_PREFIXES.items():
                if feat_name.startswith(prefix):
                    group_sums[group_name] += val
                    group_counts[group_name] += 1
                    matched = True
                    break
            if not matched:
                group_sums["WORLD MODEL THREAT FEATURES"] += val
                group_counts["WORLD MODEL THREAT FEATURES"] += 1
                
        total_val = sum(group_sums.values()) + 1e-10
        group_importance = {}
        for g_name, g_val in group_sums.items():
            group_importance[g_name] = {
                "total_shap_importance": float(g_val),
                "percentage_share": float((g_val / total_val) * 100.0),
                "feature_count": group_counts[g_name]
            }
            
        json_path = self.logs_dir / "shap_group_importance.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(group_importance, f, indent=2)
        logger.info(f"Saved feature group SHAP importance to {json_path}")
        return group_importance

    def generate_explanation(self, x_vec: np.ndarray, risk_level: str = "MODERATE", attack_prob: float = 0.5) -> str:
        """
        Generates dynamic, human-readable threat explanation summary from actual top local SHAP values.
        """
        local_contribs = self.explain_prediction(x_vec)
        top_risk_drivers = [c for c in local_contribs if c["direction"] == "INCREASES_RISK"][:3]
        
        summary_lines = [
            f"Prediction Threat Evaluation: {risk_level} RISK (Predicted Attack Likelihood: {attack_prob * 100:.1f}%).",
            "Top Influential Feature Drivers:"
        ]
        
        if top_risk_drivers:
            for i, d in enumerate(top_risk_drivers, start=1):
                summary_lines.append(f"  {i}. {d['feature']} (Value: {d['value']:.2f}, SHAP Contribution: +{d['shap_value']:.4f})")
            
            driver_names = ", ".join([d["feature"] for d in top_risk_drivers])
            summary_lines.append(f"Technical Summary: The elevated risk score is primarily driven by anomalous activity in {driver_names}.")
        else:
            summary_lines.append("  No individual feature exhibited a significant positive risk escalation contribution.")
            summary_lines.append("Technical Summary: Network metrics remain within expected normal baseline operating ranges.")
            
        return "\n".join(summary_lines)

if __name__ == "__main__":
    explainer = SHAPExplainer()
    dummy_x = np.random.randn(50, 489).astype(np.float32)
    global_imp = explainer.compute_global_shap(dummy_x, sample_size=50)
    single_exp = explainer.explain_prediction(dummy_x[0])
    text_exp = explainer.generate_explanation(dummy_x[0], risk_level="HIGH", attack_prob=0.85)
    
    print("SHAP Explainer Test Passed:")
    print(f"  - Global Features Extracted : {len(global_imp)}")
    print(f"  - Top 1 Global Feature     : {global_imp[0]['feature']} (SHAP: {global_imp[0]['mean_abs_shap']:.4f})")
    print(f"  - Local Explanation Features: {len(single_exp)}")
    print("\nGenerated Explanation Text:\n" + text_exp)
