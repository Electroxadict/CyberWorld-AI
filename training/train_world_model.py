"""
PyTorch LSTM Temporal World Model Training Engine for CyberWorld-AI.
Trains multi-task temporal world model on sequences (S[t-9]...S[t] -> S[t+1]),
applies early stopping, saves best PyTorch checkpoint (.pt) and config (.json),
and evaluates test set metrics for state reconstruction, attack detection, and stage mapping.
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.check_dataset import load_config
from models.world_model import TemporalWorldModel
from training.train_logistic import evaluate_classifier_metrics
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_world_model(config_path="config.yaml"):
    """
    Loads sequence datasets, initializes PyTorch World Model, trains with multi-task loss,
    applies early stopping, saves model checkpoint, and evaluates test set performance.
    """
    config = load_config(config_path)
    seq_dir = Path(config["paths"]["sequences_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Device Selection & Seed
    seed = config["training"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device Selection] Using compute device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Inspect and Load NumPy Sequences
    logger.info(f"Loading temporal sequence datasets from {seq_dir}...")
    
    x_train_p = seq_dir / "X_train_seq.npy"
    if not x_train_p.exists():
        from training.create_sequences import create_temporal_sequences
        logger.info("Sequence files not found. Creating temporal sequences...")
        create_temporal_sequences(config_path=config_path)
        
    X_train = np.load(seq_dir / "X_train_seq.npy")
    Y_train_state = np.load(seq_dir / "Y_train_state.npy")
    Y_train_attack = np.load(seq_dir / "Y_train_attack.npy")
    Y_train_stage = np.load(seq_dir / "Y_train_stage.npy")
    
    X_val = np.load(seq_dir / "X_val_seq.npy")
    Y_val_state = np.load(seq_dir / "Y_val_state.npy")
    Y_val_attack = np.load(seq_dir / "Y_val_attack.npy")
    Y_val_stage = np.load(seq_dir / "Y_val_stage.npy")
    
    X_test = np.load(seq_dir / "X_test_seq.npy")
    Y_test_state = np.load(seq_dir / "Y_test_state.npy")
    Y_test_attack = np.load(seq_dir / "Y_test_attack.npy")
    Y_test_stage = np.load(seq_dir / "Y_test_stage.npy")
    
    num_samples, seq_len, num_features = X_train.shape
    logger.info(f"Detected Sequence Shape -> Samples: {num_samples}, Seq Len: {seq_len}, Num Features: {num_features}")
    
    # Load feature column names list if present
    feature_cols_p = models_dir / "feature_columns.pkl"
    if feature_cols_p.exists():
        with open(feature_cols_p, "rb") as f:
            feature_cols = pickle.load(f)
    else:
        feature_cols = [f"feature_{i}" for i in range(num_features)]
        
    # 3. Create PyTorch Datasets & DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train_state, dtype=torch.float32),
        torch.tensor(Y_train_attack, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_train_stage, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val_state, dtype=torch.float32),
        torch.tensor(Y_val_attack, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_val_stage, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test_state, dtype=torch.float32),
        torch.tensor(Y_test_attack, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_test_stage, dtype=torch.long)
    )
    
    batch_size = config["training"].get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 4. Instantiate World Model
    hidden_size = config["model"].get("hidden_size", 128)
    num_layers = config["model"].get("num_layers", 2)
    dropout = config["model"].get("dropout", 0.2)
    
    model = TemporalWorldModel(
        num_features=num_features,
        embedding_size=64,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_stages=6
    ).to(device)
    
    # 5. Multi-task Loss Criteria & Weights
    lambda_state = config["model"].get("lambda_state", 0.5)
    lambda_attack = config["model"].get("lambda_attack", 0.3)
    lambda_stage = config["model"].get("lambda_stage", 0.2)
    
    criterion_state = nn.SmoothL1Loss()
    criterion_attack = nn.BCEWithLogitsLoss()
    criterion_stage = nn.CrossEntropyLoss()
    
    lr = config["training"].get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epochs = config["training"].get("epochs", 30)
    patience = config["training"].get("early_stopping_patience", 5)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = models_dir / "world_model.pt"
    
    print("\n" + "=" * 80)
    print(f" STARTING TEMPORAL WORLD MODEL TRAINING ({epochs} Epochs Max)")
    print("=" * 80)
    
    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_state_loss = 0.0
        train_attack_loss = 0.0
        train_stage_loss = 0.0
        
        for b_x, b_state, b_attack, b_stage in train_loader:
            b_x, b_state = b_x.to(device), b_state.to(device)
            b_attack, b_stage = b_attack.to(device), b_stage.to(device)
            
            optimizer.zero_grad()
            pred_state, attack_logits, stage_logits, _ = model(b_x)
            
            loss_s = criterion_state(pred_state, b_state)
            loss_a = criterion_attack(attack_logits, b_attack)
            loss_stg = criterion_stage(stage_logits, b_stage)
            
            total_loss = (lambda_state * loss_s) + (lambda_attack * loss_a) + (lambda_stage * loss_stg)
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * len(b_x)
            train_state_loss += loss_s.item() * len(b_x)
            train_attack_loss += loss_a.item() * len(b_x)
            train_stage_loss += loss_stg.item() * len(b_x)
            
        train_loss /= len(train_ds)
        train_state_loss /= len(train_ds)
        train_attack_loss /= len(train_ds)
        train_stage_loss /= len(train_ds)
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_attack_loss = 0.0
        val_stage_loss = 0.0
        
        with torch.no_grad():
            for b_x, b_state, b_attack, b_stage in val_loader:
                b_x, b_state = b_x.to(device), b_state.to(device)
                b_attack, b_stage = b_attack.to(device), b_stage.to(device)
                
                pred_state, attack_logits, stage_logits, _ = model(b_x)
                
                loss_s = criterion_state(pred_state, b_state)
                loss_a = criterion_attack(attack_logits, b_attack)
                loss_stg = criterion_stage(stage_logits, b_stage)
                
                total_loss = (lambda_state * loss_s) + (lambda_attack * loss_a) + (lambda_stage * loss_stg)
                
                val_loss += total_loss.item() * len(b_x)
                val_state_loss += loss_s.item() * len(b_x)
                val_attack_loss += loss_a.item() * len(b_x)
                val_stage_loss += loss_stg.item() * len(b_x)
                
        val_loss /= len(val_ds)
        val_state_loss /= len(val_ds)
        val_attack_loss /= len(val_ds)
        val_stage_loss /= len(val_ds)
        
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Att: {train_attack_loss:.4f}, Stg: {train_stage_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Att: {val_attack_loss:.4f}, Stg: {val_stage_loss:.4f})"
        )
        
        # Early Stopping & Checkpoint Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save PyTorch Checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_features": num_features,
                    "sequence_length": seq_len,
                    "embedding_size": 64,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "num_stages": 6
                },
                "feature_columns": feature_cols,
                "training_config": config
            }
            torch.save(checkpoint, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
                break
                
    print("=" * 80 + "\n")
    logger.info(f"Saved best model checkpoint to {best_checkpoint_path}")
    
    # Save Model Config JSON for easy inspection
    model_config_json = {
        "num_features": num_features,
        "sequence_length": seq_len,
        "embedding_size": 64,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "num_stages": 6,
        "feature_columns": feature_cols,
        "best_val_loss": float(best_val_loss)
    }
    json_path = models_dir / "model_config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_config_json, f, indent=2)
    logger.info(f"Saved model configuration JSON to {json_path}")
    
    # 7. Evaluate Best Checkpoint on Test Set
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_pred_states = []
    all_true_states = []
    all_pred_attack_probs = []
    all_true_attacks = []
    all_pred_stages = []
    all_true_stages = []
    
    with torch.no_grad():
        for b_x, b_state, b_attack, b_stage in test_loader:
            b_x = b_x.to(device)
            pred_state, attack_logits, stage_logits, _ = model(b_x)
            
            attack_probs = torch.sigmoid(attack_logits).cpu().numpy()
            stage_preds = torch.argmax(stage_logits, dim=-1).cpu().numpy()
            
            all_pred_states.append(pred_state.cpu().numpy())
            all_true_states.append(b_state.numpy())
            all_pred_attack_probs.append(attack_probs)
            all_true_attacks.append(b_attack.numpy())
            all_pred_stages.append(stage_preds)
            all_true_stages.append(b_stage.numpy())
            
    pred_states_arr = np.vstack(all_pred_states)
    true_states_arr = np.vstack(all_true_states)
    pred_attack_probs_arr = np.vstack(all_pred_attack_probs).ravel()
    true_attacks_arr = np.vstack(all_true_attacks).ravel()
    pred_stages_arr = np.concatenate(all_pred_stages)
    true_stages_arr = np.concatenate(all_true_stages)
    
    # Compute State Reconstruction Metrics (MAE, MSE, RMSE)
    mae = mean_absolute_error(true_states_arr, pred_states_arr)
    mse = mean_squared_error(true_states_arr, pred_states_arr)
    rmse = np.sqrt(mse)
    
    # Compute Attack Classification Metrics
    pred_attack_binary = (pred_attack_probs_arr >= 0.5).astype(int)
    attack_metrics = evaluate_classifier_metrics(true_attacks_arr, pred_attack_binary, pred_attack_probs_arr)
    
    # Compute Attack Stage Multi-class Metrics
    stage_acc = accuracy_score(true_stages_arr, pred_stages_arr)
    stage_f1_macro = f1_score(true_stages_arr, pred_stages_arr, average="macro", zero_division=0)
    
    print("=" * 80)
    print(" TEMPORAL WORLD MODEL TEST EVALUATION (SMOKE TEST / LOCAL DATASET)")
    print("=" * 80)
    print(" [1. Future Network State Prediction (S[t+1])]")
    print(f"     - State MAE  : {mae:.4f}")
    print(f"     - State MSE  : {mse:.4f}")
    print(f"     - State RMSE : {rmse:.4f}")
    print(" [2. Binary Attack Detection]")
    for k, v in attack_metrics.items():
        print(f"     - {k:<20}: {v:.4f}")
    print(" [3. MITRE Attack Stage Mapping]")
    print(f"     - Stage Accuracy : {stage_acc:.4f}")
    print(f"     - Stage Macro F1 : {stage_f1_macro:.4f}")
    print("=" * 80 + "\n")
    
    return model, {
        "state_mae": mae, "state_mse": mse, "state_rmse": rmse,
        "attack_metrics": attack_metrics,
        "stage_acc": stage_acc, "stage_f1_macro": stage_f1_macro
    }

if __name__ == "__main__":
    train_world_model()
