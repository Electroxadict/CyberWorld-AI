"""
Unit tests for CyberWorld-AI models, feature normalizers, and PyTorch Temporal World Model.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.normalizer import FeatureNormalizer
from models.world_model import TemporalWorldModel
from training.train_logistic import evaluate_classifier_metrics

def test_feature_normalizer():
    normalizer = FeatureNormalizer()
    X_dummy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    
    scaled_train = normalizer.fit_transform(X_dummy)
    assert scaled_train.shape == X_dummy.shape
    assert normalizer.is_fitted
    
    X_val = np.array([[2.0, 3.0]], dtype=np.float32)
    scaled_val = normalizer.transform(X_val)
    assert scaled_val.shape == X_val.shape

def test_evaluation_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = evaluate_classifier_metrics(y_true, y_pred, y_prob)
    assert "Accuracy" in metrics
    assert "F1" in metrics
    assert "False Positive Rate" in metrics
    assert "ROC-AUC" in metrics
    assert 0.0 <= metrics["Accuracy"] <= 1.0

def test_temporal_world_model_shapes():
    batch_size = 4
    seq_len = 10
    num_features = 69
    
    x = torch.randn(batch_size, seq_len, num_features)
    model = TemporalWorldModel(num_features=num_features, embedding_size=64, hidden_size=128, num_layers=2)
    
    # 1. Forward Pass
    pred_s, att_log, stg_log, attn = model(x)
    assert pred_s.shape == (batch_size, num_features)
    assert att_log.shape == (batch_size, 1)
    assert stg_log.shape == (batch_size, 6)
    assert attn.shape == (batch_size, seq_len)
    
    # 2. Predict Methods
    next_s = model.predict_next_state(x)
    assert next_s.shape == (batch_size, num_features)
    
    att_prob = model.predict_attack_probability(x)
    assert att_prob.shape == (batch_size, 1)
    assert (att_prob >= 0.0).all() and (att_prob <= 1.0).all()
    
    stg_prob = model.predict_attack_stage(x)
    assert stg_prob.shape == (batch_size, 6)
    assert torch.allclose(stg_prob.sum(dim=-1), torch.ones(batch_size))
    
    # 3. K-Step Rollout (5 steps)
    fut_states, fut_att_probs, fut_stg_probs = model.rollout_future_states(x, steps=5)
    assert fut_states.shape == (batch_size, 5, num_features)
    assert fut_att_probs.shape == (batch_size, 5, 1)
    assert fut_stg_probs.shape == (batch_size, 5, 6)
