# CyberWorld-AI Architecture Documentation

## 1. System Overview
**CyberWorld-AI** is an offline, explainable AI cybersecurity prototype designed for predictive cyber defence. Rather than acting reactively after an attack has compromised network assets, CyberWorld-AI learns how network traffic behavior evolves over time, models state transition dynamics using a **PyTorch LSTM Temporal World Model**, simulates future network trajectories through **5-step recursive rollouts**, predicts attack progression and MITRE ATT&CK stages using **XGBoost**, calculates dynamic risk scores, and provides transparent explanations using **TreeSHAP** and **Temporal Attention**.

---

## 2. Data Flow Architecture
```
Raw PCAP / Flow Telemetry
          │
          ▼
Scapy Feature Extraction (Canonical 5-Tuple Flows)
          │
          ▼
5-Second Temporal State Windows
          │
          ▼
69 Base Network State Features (Feature Engineering)
          │
          ▼
StandardScaler Normalization (Training Scaler)
          │
          ▼
10-State Temporal Sequence (S[t-9] ... S[t])
          │
          ▼
PyTorch LSTM Temporal World Model
          │
          ├─────────────────────────┬─────────────────────────┐
          ▼                         ▼                         ▼
Predicted Next State S[t+1]    Attack Logits             MITRE Stage Logits
          │
          ▼ (Recursive 5-Step Rollout)
Future State Trajectory (S[t+1] ... S[t+5])
          │
          ▼
XGBoost Feature Construction (489 Temporal Features)
          │
          ├───────────────────────────────────────────────────┐
          ▼                                                   ▼
XGBoost Risk Model                                  XGBoost MITRE Stage Model
          │                                                   │
          ▼                                                   ▼
Attack Probability                                  6-Class Stage Trajectory
          │                                                   │
          └─────────────────────────┬─────────────────────────┘
                                    ▼
                         Phase 5 Risk Engine
                                    │
                                    ▼
                        Early Warning Alert System
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
TreeSHAP Feature Driver Explanation              World Model Temporal Attention
          │                                                   │
          └─────────────────────────┬─────────────────────────┘
                                    ▼
                 Streamlit SOC Control-Room Dashboard
```

---

## 3. Feature Engineering (69 Base Features)
Network flows are aggregated into 69 flow-level and macro-level behavioral metrics per temporal window:
- **Port Statistics**: Mean, std, min, max for source and destination ports.
- **Packet & Byte Telemetry**: Forward/backward packet counts, forward/backward total bytes, mean packet lengths, flow bytes/sec, flow packets/sec.
- **TCP Flags & Windowing**: Counts for SYN, ACK, RST, FIN, PSH, URG flags, mean TTL, mean TCP window size.
- **Macro Behavioral Indicators**:
  - `Unique_Src_Ports` & `Unique_Dst_Ports`
  - `SYN_ACK_Ratio`: $\frac{\text{SYN}}{\text{ACK} + 1.0}$
  - `RST_SYN_Ratio`: $\frac{\text{RST}}{\text{SYN} + 1.0}$
  - `Port_Diversity`: $\frac{\text{Unique Dst Ports}}{\text{Flow Count} + 1.0}$
  - `Connection_Failure_Rate`: $\frac{\text{RST}}{\text{Total Packets} + 1.0}$
  - `Traffic_Burst_Score`: $\frac{\text{Current Packets}}{\text{Moving Mean Packets} + 1.0}$
  - `Port_Scan_Score`: $\frac{\text{Unique Dst Ports} \times \text{SYN}}{\text{Flow Count} + 1.0}$

---

## 4. Temporal Windowing & Sequence Creation
- **Time Window**: Fixed 5-second non-overlapping temporal windows (`time_window_seconds: 5`).
- **Sequence Length**: 10 consecutive temporal states ($S_{t-9}, S_{t-8}, \dots, S_t$), spanning 50 seconds of historical network behavior.
- **Chronological Preservation**: Sequences are strictly created chronologically without temporal shuffling to prevent data leakage.

---

## 5. PyTorch LSTM Temporal World Model
- **Encoder**: Linear layer mapping 69 features to 128 hidden dimensions.
- **Recurrent Core**: 2-layer LSTM (`hidden_size=128`, `dropout=0.2`).
- **Temporal Attention**: Learns normalized scalar weights $\alpha = [\alpha_{t-9}, \dots, \alpha_t]$ ($\sum \alpha_i = 1.0$) across historical sequence windows.
- **State Decoder Head**: Predicts $\hat{S}_{t+1} \in \mathbb{R}^{69}$.
- **Multi-Task Heads**: Binary attack classification head and 6-class MITRE ATT&CK stage head.

---

## 6. Recursive $K$-Step Rollout Simulation
- Performs 5-step forward simulation ($\hat{S}_{t+1}, \hat{S}_{t+2}, \dots, \hat{S}_{t+5}$) without using ground-truth future traffic.
- At each step $k$, the predicted state $\hat{S}_{t+k}$ is appended to the sequence buffer, sliding out $S_{t-9+k}$, to auto-regressively predict subsequent states.

---

## 7. XGBoost Risk & Stage Predictor (489 Features)
Combines current state $S_t$ (69 features) and 5-step World Model rollout statistics (345 rollout stats + 6 trajectory indicators) into a **489-dimension temporal feature vector**:
- Current state metrics (`curr_*`)
- Rollout statistics: `fut_mean_*`, `fut_max_*`, `fut_min_*`, `fut_diff_*`, `fut_pct_*`, `fut_slope_*`
- World Model indicators: `wm_current_attack_prob`, `wm_future_attack_mean`, `wm_future_attack_max`, `wm_future_attack_slope`, `wm_future_stage_max_prob`, `wm_future_stage_dominant`.

---

## 8. MITRE ATT&CK Stage Mapping
Maps predicted stage probabilities to 6 canonical attack progression stages using `MitreStageMapper`:
0. **Normal**: Benign baseline background traffic.
1. **Reconnaissance**: Port scanning, host discovery.
2. **Initial Access**: Brute-force authentication, web exploitation.
3. **Lateral Movement / DoS**: Internal spreading, resource exhaustion flooding.
4. **Command and Control**: C2 beaconing, botnet traffic.
5. **Exfiltration**: Abnormally large data transfer out of perimeter.

---

## 9. Phase 5 Risk Engine Formula
Calculates a 0–100 threat score using the verified multi-component formula:
$$\text{Risk Score} = 100 \times \left( 0.60 \times P_{\text{attack}} + 0.20 \times P_{\text{progression}} + 0.10 \times S_{\text{anomaly}} + 0.10 \times S_{\text{trend}} \right)$$
- **Categories**: `LOW` ($< 30$), `MODERATE` ($30 - 60$), `HIGH` ($60 - 80$), `CRITICAL` ($\ge 80$).

---

## 10. Early Warning Alert System
- Computes trajectory trends across the 5-step rollout horizon (+5s to +25s).
- Calculates $T_{\text{high\_risk}}$ (estimated seconds until risk exceeds 60.0).
- Emits actionable security advisories before high-risk progression occurs.

---

## 11. TreeSHAP Explainability Engine
- Uses `shap.TreeExplainer` on the 489-feature XGBoost risk model.
- Provides local feature contributions ($+$ increases risk / $-$ decreases risk).
- Groups features into 8 macro categories (`CURRENT`, `FUTURE MEAN`, `FUTURE MAX`, `FUTURE MIN`, `FUTURE DIFFERENCE`, `FUTURE PERCENTAGE CHANGE`, `FUTURE SLOPE`, `WORLD MODEL THREAT FEATURES`).
- Includes version compatibility patches for XGBoost 3.x string formatting.

---

## 12. World Model Temporal Attention Visualizer
- Extracts attention weights $\alpha \in \mathbb{R}^{10}$ from the PyTorch World Model.
- Identifies the most influential historical timestep (e.g. $t-3$, ~15 seconds prior).
- Generates human-readable temporal explanation text.

---

## 13. PCAP Ingestion Pipeline (`PCAPPredictivePipeline`)
- Parses `.pcap` and `.pcapng` files using Scapy.
- Constructs canonical 5-tuple flows `(src_ip, dst_ip, src_port, dst_port, proto)` with layer defense checks.
- Validates 69-feature schema matching and applies existing training `StandardScaler` without refitting.

---

## 14. Streamlit Control Room Dashboard (`app.py`)
- Dark SOC control-room layout with 5 interactive tabs.
- Real-time Plotly Risk Gauges, Future Risk line forecasts, Attack Probability progression, SHAP bar charts, and Temporal Attention charts.
- JSON analysis exporter (`st.download_button`).

---

## 15. Limitations & Defensive Security Scope
1. **Research Prototype**: Designed as an offline research prototype for decision support.
2. **Data Dependency**: Prediction accuracy depends on training data quality and baseline normalization.
3. **Defensive Scope Only**: Performs traffic parsing, classification, prediction, risk scoring, and explainability. Includes no offensive capabilities.
