"""
End-to-End PCAP Predictive Cyber Defence Pipeline for CyberWorld-AI.
Connects raw PCAP packet parsing to 5-second temporal windowing, schema validation,
trained feature scalers, PyTorch World Model 5-step rollouts, XGBoost risk/stage classifiers,
Risk Engine scoring, Early Warning alerts, SHAP feature contributions, and Temporal Attention visualizations.
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

from preprocessing.check_dataset import load_config
from preprocessing.feature_extractor import PCAPFeatureExtractor
from preprocessing.time_window import RESERVED_COLUMNS
from preprocessing.preprocess import clean_dataframe
from preprocessing.normalizer import FeatureNormalizer
from inference.predictor import WorldModelPredictor
from inference.rollout import RolloutSimulator
from inference.risk_engine import RiskEngine
from inference.early_warning import EarlyWarningEngine
from inference.xgboost_predictor import XGBoostPredictor
from explainability.shap_explainer import SHAPExplainer
from explainability.attention_visualizer import TemporalAttentionVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PCAPPredictivePipeline:
    """End-to-End PCAP Predictive Cyber Defence Pipeline."""
    
    def __init__(self, models_dir=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        if models_dir is None:
            self.models_dir = Path(self.config["paths"]["models_dir"])
        else:
            self.models_dir = Path(models_dir)
            
        self.window_sec = self.config.get("temporal", {}).get("time_window_seconds", 5)
        self.seq_len = self.config.get("temporal", {}).get("sequence_length", 10)
        self.horizon_steps = self.config.get("temporal", {}).get("prediction_horizon", 5)
        
        self.feature_extractor = PCAPFeatureExtractor(config_path=config_path)
        self.feature_columns = []
        self.scaler = None
        
        self.predictor = None
        self.rollout_sim = None
        self.risk_engine = None
        self.early_warning = None
        self.xgb_predictor = None
        self.shap_explainer = None
        self.attn_visualizer = None
        
        self.load_models()

    def load_models(self):
        """Loads all trained model artifacts, scalers, feature column schemas, and explainers."""
        cols_path = self.models_dir / "feature_columns.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        
        if not cols_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Required scaler artifacts not found in {self.models_dir}. Run preprocessing and model training first.")
            
        with open(cols_path, "rb") as f:
            self.feature_columns = pickle.load(f)
            
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        logger.info(f"Loaded training feature column schema ({len(self.feature_columns)} features) and StandardScaler.")
        
        self.predictor = WorldModelPredictor(models_dir=self.models_dir)
        self.rollout_sim = RolloutSimulator(predictor=self.predictor)
        self.risk_engine = RiskEngine(models_dir=self.models_dir)
        self.early_warning = EarlyWarningEngine(predictor=self.predictor, rollout_sim=self.rollout_sim, risk_engine=self.risk_engine)
        self.xgb_predictor = XGBoostPredictor(models_dir=self.models_dir)
        self.shap_explainer = SHAPExplainer(models_dir=self.models_dir)
        self.attn_visualizer = TemporalAttentionVisualizer(predictor=self.predictor)

    def extract_features(self, pcap_path, max_packets=None) -> pd.DataFrame:
        """Parses PCAP into flow records."""
        return self.feature_extractor.extract(pcap_path, max_packets=max_packets)

    def create_temporal_windows(self, df_flows: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates raw PCAP flow records into 5-second temporal state windows.
        Matches exact feature semantics and ordering of training pipeline.
        """
        df_clean = clean_dataframe(df_flows)
        
        min_time = df_clean["Timestamp"].min()
        df_clean["Window_ID"] = ((df_clean["Timestamp"] - min_time).dt.total_seconds() // self.window_sec).astype(int)
        
        grouped = df_clean.groupby("Window_ID")
        window_records = []
        
        flow_num_cols = [c for c in df_clean.columns if c not in RESERVED_COLUMNS and pd.api.types.is_numeric_dtype(df_clean[c])]
        window_pkts_history = []
        
        for window_id, group in grouped:
            record = {}
            record["Window_ID"] = window_id
            record["Window_Timestamp"] = group["Timestamp"].min()
            record["Flow_Count"] = len(group)
            
            for col in flow_num_cols:
                vals = group[col]
                record[f"{col}_mean"] = vals.mean()
                record[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0
                record[f"{col}_max"] = vals.max()
                record[f"{col}_min"] = vals.min()
                
            for k in list(record.keys()):
                if isinstance(record[k], float) and np.isnan(record[k]):
                    record[k] = 0.0
                    
            src_port_col = [c for c in group.columns if "src" in c.lower() and "port" in c.lower()]
            dst_port_col = [c for c in group.columns if "dst" in c.lower() and "port" in c.lower()]
            
            unique_src_ports = group[src_port_col[0]].nunique() if src_port_col else 1
            unique_dst_ports = group[dst_port_col[0]].nunique() if dst_port_col else 1
            
            syn_cnt = group["SYN Flag Cnt"].sum() if "SYN Flag Cnt" in group.columns else 0
            ack_cnt = group["ACK Flag Cnt"].sum() if "ACK Flag Cnt" in group.columns else 0
            rst_cnt = group["RST Flag Cnt"].sum() if "RST Flag Cnt" in group.columns else 0
            
            tot_pkts = (group["Tot Fwd Pkts"].sum() + group["Tot Bwd Pkts"].sum()) if "Tot Fwd Pkts" in group.columns and "Tot Bwd Pkts" in group.columns else len(group)
            
            record["Unique_Src_Ports"] = unique_src_ports
            record["Unique_Dst_Ports"] = unique_dst_ports
            record["SYN_ACK_Ratio"] = float(syn_cnt) / (float(ack_cnt) + 1.0)
            record["RST_SYN_Ratio"] = float(rst_cnt) / (float(syn_cnt) + 1.0)
            record["Port_Diversity"] = float(unique_dst_ports) / (float(len(group)) + 1.0)
            record["Connection_Failure_Rate"] = float(rst_cnt) / (float(tot_pkts) + 1.0)
            
            window_pkts_history.append(tot_pkts)
            moving_mean_pkts = np.mean(window_pkts_history[-5:])
            record["Traffic_Burst_Score"] = float(tot_pkts) / (float(moving_mean_pkts) + 1.0)
            record["Port_Scan_Score"] = float(unique_dst_ports * syn_cnt) / (float(len(group)) + 1.0)
            
            record["Attack"] = 1 if (group["Attack"] == 1).any() else 0
            record["Attack_Stage"] = int(group["Attack_Stage"].max()) if "Attack_Stage" in group.columns else 0
            
            window_records.append(record)
            
        window_df = pd.DataFrame(window_records).sort_values("Window_ID").reset_index(drop=True)
        return window_df

    def validate_and_scale_features(self, window_df: pd.DataFrame) -> np.ndarray:
        """
        Validates feature columns against persisted training schema and applies existing scaler.
        """
        missing = [c for c in self.feature_columns if c not in window_df.columns]
        if missing:
            # Fill missing features with 0.0
            logger.warning(f"Filling {len(missing)} missing feature columns with default 0.0: {missing[:5]}...")
            for col in missing:
                window_df[col] = 0.0
                
        # Select exact persisted feature column order
        X_df = window_df[self.feature_columns]
        
        if X_df.shape[1] != len(self.feature_columns):
            raise ValueError(f"Feature count mismatch: Got {X_df.shape[1]}, expected {len(self.feature_columns)}")
            
        # Transform using existing fitted scaler (DO NOT FIT NEW SCALER!)
        scaled_matrix = self.scaler.transform(X_df.values)
        return scaled_matrix

    def create_sequence(self, scaled_matrix: np.ndarray) -> np.ndarray:
        """
        Constructs sequence window of shape (1, 10, 69) from scaled time windows.
        Raises human-readable error if history is fewer than 10 windows.
        """
        num_windows = len(scaled_matrix)
        if num_windows < self.seq_len:
            raise ValueError(
                f"Insufficient temporal history in PCAP. Available temporal windows: {num_windows}, "
                f"Required history for World Model: {self.seq_len}. PCAP must span at least {self.seq_len * self.window_sec} seconds."
            )
            
        # Select most recent 10 consecutive temporal windows
        recent_seq = scaled_matrix[-self.seq_len:] # (10, 69)
        seq_tensor_np = np.expand_dims(recent_seq, axis=0) # (1, 10, 69)
        return seq_tensor_np

    def predict(self, pcap_path, max_packets=None) -> dict:
        """
        Executes complete end-to-end predictive pipeline for a PCAP file.
        """
        pcap_p = Path(pcap_path)
        logger.info(f"--- Starting CyberWorld-AI Predictive Defence for PCAP: {pcap_p.name} ---")
        
        # 1. Scapy Feature Extraction
        df_flows = self.extract_features(pcap_p, max_packets=max_packets)
        
        # 2. 5-Second Window Aggregation
        df_windows = self.create_temporal_windows(df_flows)
        
        # 3. Schema Validation & Feature Scaling
        scaled_matrix = self.validate_and_scale_features(df_windows)
        
        # 4. Temporal Sequence Construction
        x_seq = self.create_sequence(scaled_matrix)
        
        # 5. Early Warning & Multi-step Forecast Analysis
        ew_analysis = self.early_warning.analyze_sequence(x_seq)
        
        # 6. XGBoost Inference & MITRE Stage Prediction
        xgb_res = self.xgb_predictor.predict(x_seq)
        
        # 7. SHAP Feature Contribution Analysis
        xgb_feat_2d = self.xgb_predictor.build_features(x_seq)
        local_shap = self.shap_explainer.explain_prediction(xgb_feat_2d[0])
        group_shap = self.shap_explainer.group_shap_features(xgb_feat_2d)
        shap_text = self.shap_explainer.generate_explanation(
            xgb_feat_2d[0],
            risk_level=ew_analysis["risk_level"],
            attack_prob=xgb_res["attack_probability"]
        )
        
        # 8. Temporal Attention Analysis
        attn_dict = self.attn_visualizer.get_temporal_attention(x_seq)
        max_label, max_idx, max_weight = self.attn_visualizer.get_most_influential_timestep(x_seq)
        attn_text = self.attn_visualizer.generate_temporal_explanation(x_seq)
        
        return {
            "source": "PCAP",
            "pcap_file": pcap_p.name,
            "flow_count": len(df_flows),
            "window_count": len(df_windows),
            "current_attack_probability": float(xgb_res["attack_probability"]),
            "risk_score": float(ew_analysis["current_risk"]),
            "risk_level": ew_analysis["risk_level"],
            "future_risk": ew_analysis["future_risks"],
            "future_attack_probability": ew_analysis["future_attack_probabilities"],
            "predicted_stage": int(xgb_res["predicted_stage"]),
            "stage_name": str(xgb_res["stage_name"]),
            "stage_probabilities": [float(p) for p in xgb_res["stage_probabilities"]],
            "time_to_high_risk": ew_analysis["time_to_high_risk"],
            "warning_triggered": ew_analysis["warning_triggered"],
            "warning_message": ew_analysis["warning_message"],
            "top_shap_features": local_shap[:10],
            "feature_group_importance": group_shap,
            "shap_explanation": shap_text,
            "attention_weights": attn_dict["attention_weights"],
            "most_influential_timestep": max_label,
            "most_influential_weight": float(max_weight),
            "temporal_explanation": attn_text
        }

if __name__ == "__main__":
    logger.info("PCAPPredictivePipeline ready.")
