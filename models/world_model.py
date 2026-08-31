"""
PyTorch LSTM Temporal World Model for CyberWorld-AI.
Learns network state dynamics P(S[t+1] | S[t], ..., S[t-n]), reconstructs future network states,
predicts attack probability and MITRE attack stages, and supports recursive K-step forward simulation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalWorldModel(nn.Module):
    """
    Temporal World Model Architecture:
    1. Feature Encoder: Project input features to embedding space.
    2. Temporal Core: 2-Layer LSTM modeling temporal network dynamics.
    3. Attention Layer: Calculates temporal window importance weights.
    4. Future State Decoder: Reconstructs next network state S[t+1].
    5. Attack Head: Classifies binary attack probability.
    6. Stage Head: Classifies MITRE ATT&CK stage (6 classes).
    """
    
    def __init__(
        self,
        num_features: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_stages: int = 6
    ):
        super().__init__()
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stages = num_stages
        
        # 1. Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Temporal Core (2-Layer LSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Temporal Attention Mechanism
        self.attn_query = nn.Linear(hidden_size, 1, bias=False)
        
        # 4. Future State Decoder Head (predicts S[t+1])
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_features)
        )
        
        # 5. Attack Probability Classification Head (logits)
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 6. Attack Stage Classification Head (logits)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_stages)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for Temporal World Model.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, sequence_length, num_features).
            
        Returns:
            tuple: (predicted_state, attack_logits, stage_logits, attn_weights)
                - predicted_state: (batch_size, num_features)
                - attack_logits: (batch_size, 1)
                - stage_logits: (batch_size, num_stages)
                - attn_weights: (batch_size, sequence_length)
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode features across each sequence time step
        # x_encoded shape: (batch_size, seq_len, embedding_size)
        x_encoded = self.encoder(x)
        
        # LSTM Temporal Pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x_encoded)
        
        # Calculate Temporal Attention Weights
        # attn_scores shape: (batch_size, seq_len, 1)
        attn_scores = self.attn_query(lstm_out)
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=-1) # (batch_size, seq_len)
        
        # Weighted context vector c shape: (batch_size, hidden_size)
        c = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Output Heads
        predicted_state = self.state_decoder(c)
        attack_logits = self.attack_head(c)
        stage_logits = self.stage_head(c)
        
        return predicted_state, attack_logits, stage_logits, attn_weights

    def predict_next_state(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts single step future network state S[t+1]."""
        self.eval()
        with torch.no_grad():
            pred_state, _, _, _ = self.forward(x)
        return pred_state

    def predict_attack_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts attack probability (0.0 to 1.0) for sequence."""
        self.eval()
        with torch.no_grad():
            _, attack_logits, _, _ = self.forward(x)
            attack_probs = torch.sigmoid(attack_logits)
        return attack_probs

    def predict_attack_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts MITRE attack stage probability distribution across 6 classes."""
        self.eval()
        with torch.no_grad():
            _, _, stage_logits, _ = self.forward(x)
            stage_probs = F.softmax(stage_logits, dim=-1)
        return stage_probs

    def rollout_future_states(self, x: torch.Tensor, steps: int = 5):
        """
        Performs genuine recursive K-step forward simulation (rollout).
        Given initial sequence S[t-9] ... S[t], predicts S[t+1], appends it, and predicts S[t+2] ... S[t+k].
        Does NOT use ground truth future states.
        
        Args:
            x (torch.Tensor): Initial sequence window of shape (batch_size, seq_len, num_features).
            steps (int): Prediction horizon K (default 5 steps).
            
        Returns:
            tuple: (future_states, future_attack_probs, future_stage_probs)
                - future_states: (batch_size, steps, num_features)
                - future_attack_probs: (batch_size, steps, 1)
                - future_stage_probs: (batch_size, steps, num_stages)
        """
        self.eval()
        current_seq = x.clone()
        batch_size, seq_len, num_features = current_seq.shape
        
        future_states = []
        future_attack_probs = []
        future_stage_probs = []
        
        with torch.no_grad():
            for step in range(steps):
                # Predict next state and threat metrics from current sequence window
                pred_state, attack_logits, stage_logits, _ = self.forward(current_seq)
                
                attack_prob = torch.sigmoid(attack_logits)
                stage_prob = F.softmax(stage_logits, dim=-1)
                
                future_states.append(pred_state.unsqueeze(1))
                future_attack_probs.append(attack_prob.unsqueeze(1))
                future_stage_probs.append(stage_prob.unsqueeze(1))
                
                # Shift sequence window left by 1 step and append predicted next state
                # current_seq[:, 1:, :] -> (batch, seq_len-1, num_features)
                # pred_state.unsqueeze(1) -> (batch, 1, num_features)
                current_seq = torch.cat([current_seq[:, 1:, :], pred_state.unsqueeze(1)], dim=1)
                
        future_states_tensor = torch.cat(future_states, dim=1)           # (batch, steps, num_features)
        future_attack_tensor = torch.cat(future_attack_probs, dim=1)     # (batch, steps, 1)
        future_stage_tensor = torch.cat(future_stage_probs, dim=1)       # (batch, steps, num_stages)
        
        return future_states_tensor, future_attack_tensor, future_stage_tensor

if __name__ == "__main__":
    # Model architecture verification
    dummy_input = torch.randn(4, 10, 69)
    model = TemporalWorldModel(num_features=69)
    pred_s, att_log, stg_log, attn = model(dummy_input)
    print("Architecture Check Passed:")
    print(f"  - Input shape          : {dummy_input.shape}")
    print(f"  - Predicted state shape: {pred_s.shape}")
    print(f"  - Attack logits shape  : {att_log.shape}")
    print(f"  - Stage logits shape   : {stg_log.shape}")
    print(f"  - Attention shape      : {attn.shape}")
    
    roll_s, roll_att, roll_stg = model.rollout_future_states(dummy_input, steps=5)
    print("5-Step Rollout Check Passed:")
    print(f"  - Future states shape  : {roll_s.shape}")
    print(f"  - Future attack shape  : {roll_att.shape}")
    print(f"  - Future stage shape   : {roll_stg.shape}")
