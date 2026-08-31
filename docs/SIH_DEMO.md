# CyberWorld-AI — SIH Presentation & Live Demo Guide

## 1. Problem Statement
Modern Cyber Security Operations Centers (SOCs) operate reactively. Traditional Intrusion Detection Systems (IDS) trigger alerts only after an attack vector has already executed payload operations on a target asset, leaving security analysts with zero lead time to prevent data breach or privilege escalation.

---

## 2. Existing Limitations
- **Reactive Detection**: Legacy tools flag anomalies after damage is done.
- **No Temporal Prediction**: Standard classifiers lack world models to forecast future network state evolution.
- **Black-Box AI**: Complex deep learning models fail to explain *why* a specific network sequence is risky or *which past time window* triggered the threat.

---

## 3. CyberWorld-AI Solution
**CyberWorld-AI** introduces an **Explainable Temporal World Model for Predictive Cyber Defence**:
- Learns non-linear network state transition dynamics.
- Forecasts future network states 25 seconds ahead using **5-step recursive rollouts**.
- Predicts attack progression probabilities & 6 MITRE ATT&CK stages using **XGBoost**.
- Generates **Early Warning Advisories** with estimated time-to-high-risk.
- Provides transparent technical explanations via **TreeSHAP** and **PyTorch Temporal Attention**.

---

## 4. 16-Point SIH Live Demonstration Checklist

- [x] **1. Environment Integrity**: `python scripts/check_environment.py` returns ALL PASS.
- [x] **2. Model Artifact Integrity**: `python scripts/validate_models.py` verifies PyTorch, XGBoost, and Scaler artifacts.
- [x] **3. PCAP Upload & Parsing**: Supports `.pcap` and `.pcapng` files via Scapy.
- [x] **4. Layer Defense**: Handles IP, IPv6, TCP, UDP, ICMP, and non-IP frames without crashing.
- [x] **5. Flow Aggregation**: Constructs canonical 5-tuple bidirectional flows `(src_ip, dst_ip, src_port, dst_port, proto)`.
- [x] **6. 5-Second Temporal Windows**: Aggregates flows into 69 network state features.
- [x] **7. 10-State Sequence Input**: Constructs $S_{t-9}\dots S_t$ sequence inputs (50s historical context).
- [x] **8. PyTorch LSTM World Model**: Computes state transitions and hidden representations.
- [x] **9. 5-Step Recursive Rollout**: Simulates future states $+5\text{s}, +10\text{s}, +15\text{s}, +20\text{s}, +25\text{s}$ ahead.
- [x] **10. XGBoost Risk Classification**: Predicts attack likelihood across 489 temporal features.
- [x] **11. MITRE Attack Stage Mapping**: Classifies traffic into 6 stages (Normal to Exfiltration).
- [x] **12. Risk Engine & Score**: Computes 0–100 risk score and level (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
- [x] **13. Early Warning Advisory**: Calculates $T_{\text{high\_risk}}$ lead time and issues alerts.
- [x] **14. TreeSHAP Explainability**: Highlights top feature drivers ($+$ increases risk / $-$ decreases risk).
- [x] **15. Temporal Attention**: Visualizes historical timestep weights ($t-9\dots t$) and identifies most influential past window.
- [x] **16. Streamlit SOC Control Room**: Interactive dark-themed UI with Plotly risk gauges, forecasts, and JSON export.

---

## 5. Live Demo Execution Commands

### A. Quick CLI Demo (2 Minutes)
```cmd
python scripts/demo.py
```

### B. Master Pipeline CLI (With SHAP & JSON Export)
```cmd
python scripts/run_pipeline.py --pcap data/raw/sample_test.pcap --horizon 5 --explain --json
```

### C. Streamlit Control Room Dashboard (Interactive Demo)
```cmd
streamlit run app.py
```

---

## 6. SIH Presentation Pitch Script (3 Minutes)

1. **Introduction (30s)**:
   > "Good morning, respected judges. We present CyberWorld-AI: an Explainable Temporal World Model for Predictive Cyber Defence. Current SOCs operate reactively — they detect threats after an attack strikes. CyberWorld-AI changes the paradigm from reactive detection to proactive future-state prediction."

2. **Core AI Architecture (60s)**:
   > "Our system processes network PCAP telemetry into 5-second temporal state windows with 69 flow and macro behavioral features. A 2-layer PyTorch LSTM World Model with temporal attention projects network behavior 25 seconds into the future using 5-step recursive rollouts. XGBoost evaluates these projected trajectories across 489 temporal features to predict attack risk and MITRE ATT&CK stages."

3. **Explainability & Early Warning (60s)**:
   > "Crucially, CyberWorld-AI is not a black box. TreeSHAP explains *why* the model predicted a risk score by exposing positive and negative feature drivers. PyTorch Temporal Attention explains *when* past anomalous traffic occurred, pinpointing the exact historical window that triggered the forecast. Our Early Warning Engine provides security analysts with estimated lead time to high risk."

4. **Live Dashboard & Conclusion (30s)**:
   > "Here is our live Streamlit SOC Control Room Dashboard. Analysts can select or upload PCAP files, view real-time risk gauges, track future risk forecasts, inspect SHAP drivers, and export structured JSON reports. CyberWorld-AI provides predictive lead time to defend network infrastructure before attacks escalate."
