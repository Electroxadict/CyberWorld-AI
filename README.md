# CyberWorld-AI: Explainable Temporal World Model for Predictive Cyber Defence

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**CyberWorld-AI** is an offline, open-source AI cybersecurity prototype designed for predictive cyber defence. Rather than reactively detecting threats after an attack strikes, CyberWorld-AI models temporal network traffic dynamics using a **PyTorch LSTM Temporal World Model**, projects future network state trajectories 25 seconds ahead using **5-step recursive rollouts**, classifies attack risk & MITRE ATT&CK stages using **XGBoost**, issues **Early Warning Advisories**, and provides transparent explanations using **TreeSHAP** and **PyTorch Temporal Attention**.

---

## 🏛️ System Architecture

```
Raw PCAP / Flow Telemetry
          │
          ▼
Scapy Feature Extraction (Canonical 5-Tuple Flows)
          │
          ▼
5-Second Temporal State Windows (69 Base Features)
          │
          ▼
StandardScaler Normalization (Training Scaler)
          │
          ▼
10-State Temporal Sequence S[t-9] ... S[t] (50s Context)
          │
          ▼
PyTorch LSTM Temporal World Model (2-Layer LSTM + Attention)
          │
          ▼ (Recursive 5-Step Rollout)
Future State Trajectory S[t+1] ... S[t+5] (+25s Horizon)
          │
          ▼
XGBoost Risk & MITRE Stage Classifiers (489 Features)
          │
          ▼
Phase 5 Risk Engine (0-100 Score) & Early Warning Engine
          │
          ├───────────────────────────────────────────────────┐
          ▼                                                   ▼
TreeSHAP Feature Driver Explanation              World Model Temporal Attention
          │                                                   │
          └─────────────────────────┬─────────────────────────┘
                                    ▼
                 Streamlit SOC Control-Room Dashboard
```

---

## ✨ Key Features

1. **Scapy PCAP Ingestion Engine**: Parses `.pcap` and `.pcapng` files into canonical 5-tuple bidirectional flow records with defensive layer checks.
2. **69 Network State Representation**: Aggregates flow telemetry, TCP windowing, flag ratios, port diversity, burst scores, and scan scores into 5-second temporal state vectors.
3. **PyTorch LSTM Temporal World Model**: 2-layer LSTM (`hidden_size=128`) predicting future network state transitions.
4. **5-Step Recursive Rollout Simulation**: Projects network states $+5\text{s}, +10\text{s}, +15\text{s}, +20\text{s}, +25\text{s}$ into the future without ground-truth future data.
5. **XGBoost Temporal Classifiers**: Evaluates 489 temporal features combining current state & rollout statistics to predict attack probability and 6 MITRE ATT&CK stages (`Normal`, `Reconnaissance`, `Initial Access`, `Lateral Movement`, `Command & Control`, `Exfiltration`).
6. **Risk Engine & Early Warning System**: Calculates 0–100 risk score and estimates time-to-high-risk ($T_{\text{high\_risk}}$).
7. **Explainable AI (XAI)**:
   - **TreeSHAP**: Identifies top feature drivers ($+$ increases risk / $-$ decreases risk) and aggregates importance into 8 macro categories.
   - **PyTorch Temporal Attention**: Pinpoints exact historical timesteps ($t-9\dots t$) that influenced the World Model forecast.
8. **Streamlit Control Room Dashboard**: Dark SOC-themed interactive monitoring interface with Plotly risk gauges, future trajectory line charts, and JSON export.

---

## 📁 Project Structure

```
CyberWorld-AI/
├── app.py                      # Main Streamlit Control Room Dashboard
├── config.yaml                 # Central System Configuration
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
│
├── attack_mapping/             # MITRE ATT&CK Stage Mapping
│   └── mitre_mapper.py
├── dashboard/                  # Streamlit UI Components & Plotly Charts
│   ├── charts.py
│   └── components.py
├── data/                       # Datasets & Sample PCAP Test Fixtures
│   └── raw/sample_test.pcap
├── docs/                       # Architecture & Presentation Guides
│   ├── architecture.md
│   └── SIH_DEMO.md
├── explainability/             # TreeSHAP & Temporal Attention Visualizers
│   ├── attention_visualizer.py
│   └── shap_explainer.py
├── inference/                  # Predictive Defence & PCAP Pipeline
│   ├── early_warning.py
│   ├── pcap_pipeline.py
│   ├── predictor.py
│   ├── risk_engine.py
│   ├── rollout.py
│   └── xgboost_predictor.py
├── logs/                       # JSON Reports & XAI Plots
│   ├── final_evaluation.json
│   ├── latest_analysis.json
│   └── *.png
├── models/                     # Trained PyTorch & XGBoost Model Artifacts
│   ├── world_model.pt
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── xgb_risk_model.pkl
│   └── xgb_stage_model.pkl
├── preprocessing/              # Flow Extraction & 5s Window Aggregation
├── scripts/                    # Master CLI Tools & Live Demo Executables
│   ├── check_environment.py
│   ├── validate_models.py
│   ├── run_pipeline.py
│   ├── analyze_pcap.py
│   ├── generate_sample_pcap.py
│   └── demo.py
└── tests/                      # PyTest Unit & Integration Suite
```

---

## ⚙️ Installation & Setup

1. **Clone Repository / Navigate to Project Directory**:
   ```cmd
   cd D:\SIH\CyberWorld-AI
   ```

2. **Create & Activate Virtual Environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Open-Source Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Verify Environment Setup**:
   ```cmd
   python scripts/check_environment.py
   ```

---

## 🚀 Execution Commands

### 1. Environment & Model Artifact Verification
```cmd
python scripts/check_environment.py
python scripts/validate_models.py
```

### 2. Live 3-Minute SIH Demonstration
```cmd
python scripts/demo.py
```

### 3. Master Pipeline CLI (With SHAP & JSON Export)
```cmd
python scripts/run_pipeline.py --pcap data/raw/sample_test.pcap --horizon 5 --explain --json
```

### 4. Interactive Streamlit SOC Dashboard
```cmd
streamlit run app.py
```

### 5. Automated Test Suite (PyTest)
```cmd
python -m pytest
```

---

## 📊 Model Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | 88.5% | 86.2% | 84.1% | 85.1% | 0.921 |
| **Random Forest Baseline** | 94.2% | 93.1% | 92.5% | 92.8% | 0.978 |
| **XGBoost Temporal Risk Model** | **96.8%** | **96.1%** | **95.4%** | **95.8%** | **0.991** |
| **XGBoost MITRE Stage Model** | 94.5% (Macro F1: 93.8%) | — | — | — | — |
| **PyTorch LSTM World Model** | MAE: 0.0412 | MSE: 0.0038 | RMSE: 0.0616 | — | — |

*Detailed metrics exported in `logs/final_evaluation.json`.*

---

## ⚠️ Research Limitations & Defensive Scope

1. **Research Prototype**: CyberWorld-AI is an offline research prototype intended for decision support and predictive monitoring.
2. **Defensive Scope Only**: Performs traffic parsing, classification, prediction, risk scoring, and explainability. Includes no offensive capabilities.
3. **World Model Trajectory**: Future rollouts project probable traffic behavior based on learned patterns; they do not guarantee deterministic outcomes.
4. **Synthetic Test Fixture Notice**: `data/raw/sample_test.pcap` is a synthetic fixture generated via Scapy for local testing. Results reflect synthetic flow attributes.
5. **MITRE Mapping**: MITRE ATT&CK stage mappings represent an engineering interpretation layer, not ground-truth threat actor attribution.
