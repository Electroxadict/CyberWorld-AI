"""
Data Feature Normalizer and Scaler Manager for CyberWorld-AI.
Fits StandardScaler ONLY on training split to prevent data leakage,
saves scaler and feature list artifacts, and provides inference scaling utils.
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
from sklearn.preprocessing import StandardScaler

from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """Manages feature normalization using StandardScaler to guarantee zero data leakage."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
    def fit_transform(self, X_train, feature_cols=None):
        """
        Fits StandardScaler strictly on X_train and returns scaled X_train.
        
        Args:
            X_train (pd.DataFrame / np.ndarray): Training features.
            feature_cols (list, optional): List of feature column names.
            
        Returns:
            np.ndarray: Scaled training feature matrix.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = list(X_train.columns)
            X_train_vals = X_train.values
        else:
            self.feature_columns = feature_cols if feature_cols is not None else [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train_vals = X_train
            
        logger.info(f"Fitting StandardScaler on {X_train_vals.shape[0]} training samples across {len(self.feature_columns)} features...")
        X_scaled = self.scaler.fit_transform(X_train_vals)
        self.is_fitted = True
        return X_scaled
        
    def transform(self, X):
        """
        Transforms validation, test, or inference data using the PREVIOUSLY FITTED scaler.
        Does NOT alter mean or variance fitted from training set.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted on training data before calling transform().")
            
        if isinstance(X, pd.DataFrame):
            # Ensure correct column ordering matching fitted features
            if self.feature_columns:
                X = X[self.feature_columns]
            X_vals = X.values
        else:
            X_vals = X
            
        return self.scaler.transform(X_vals)
        
    def save(self, models_dir=None, scaler_filename="scaler.pkl", cols_filename="feature_columns.pkl"):
        """Saves fitted scaler and feature list to pickle files."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer.")
            
        if models_dir is None:
            config = load_config()
            models_dir = Path(config["paths"]["models_dir"])
        else:
            models_dir = Path(models_dir)
            
        models_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_path = models_dir / scaler_filename
        cols_path = models_dir / cols_filename
        
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
            
        with open(cols_path, "wb") as f:
            pickle.dump(self.feature_columns, f)
            
        logger.info(f"Saved fitted scaler to {scaler_path}")
        logger.info(f"Saved feature column names list to {cols_path}")
        return scaler_path, cols_path
        
    def load(self, models_dir=None, scaler_filename="scaler.pkl", cols_filename="feature_columns.pkl"):
        """Loads fitted scaler and feature list from pickle files."""
        if models_dir is None:
            config = load_config()
            models_dir = Path(config["paths"]["models_dir"])
        else:
            models_dir = Path(models_dir)
            
        scaler_path = models_dir / scaler_filename
        cols_path = models_dir / cols_filename
        
        if not scaler_path.exists() or not cols_path.exists():
            raise FileNotFoundError(f"Scaler artifacts not found in {models_dir}")
            
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        with open(cols_path, "rb") as f:
            self.feature_columns = pickle.load(f)
            
        self.is_fitted = True
        logger.info(f"Loaded scaler and {len(self.feature_columns)} feature columns from {models_dir}")
        return self

if __name__ == "__main__":
    logger.info("Normalizer module ready.")
