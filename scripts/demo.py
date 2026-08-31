"""
SIH Live Demonstration Script for CyberWorld-AI.
Executes 2-5 minute live demonstration of the Explainable Temporal World Model
on sample PCAP network traffic and displays formatted SOC control-room findings.

Usage: python scripts/demo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from inference.pcap_pipeline import PCAPPredictivePipeline
from scripts.generate_sample_pcap import generate_sample_pcap

def run_sih_demo():
    print("\n" + "=" * 65)
    print("      CYBERWORLD-AI — LIVE SIH DEMONSTRATION")
    print("  Explainable Temporal World Model for Predictive Cyber Defence")
    print("=" * 65)
    
    sample_pcap = PROJECT_ROOT / "data" / "raw" / "sample_test.pcap"
    if not sample_pcap.exists():
        print("[INFO] Generating synthetic test PCAP file (60 seconds of traffic)...")
        generate_sample_pcap(sample_pcap)
        
    print(f"\n[STEP 1/5] Ingesting Network PCAP Telemetry ({sample_pcap.name})...")
    time.sleep(0.5)
    
    pipeline = PCAPPredictivePipeline()
    print("[STEP 2/5] Aggregating 5-second Temporal Windows & 69 Features...")
    time.sleep(0.5)
    
    print("[STEP 3/5] Running PyTorch LSTM Temporal World Model 5-Step Rollout...")
    time.sleep(0.5)
    
    print("[STEP 4/5] Computing XGBoost Risk Classification & MITRE Attack Stage...")
    time.sleep(0.5)
    
    print("[STEP 5/5] Extracting SHAP Feature Drivers & Temporal Window Attention...")
    start_t = time.time()
    res = pipeline.predict(sample_pcap)
    elapsed = time.time() - start_t
    
    print("\n" + "=" * 65)
    print("                   LIVE DEMO ANALYSIS RESULT")
    print("=" * 65)
    print(f"PCAP Traffic File           : {res['pcap_file']}")
    print(f"Extracted Flows / Windows   : {res['flow_count']} flows across {res['window_count']} temporal windows")
    print(f"Analyzed Sequence Window    : 10 states (50 seconds history)")
    print(f"Pipeline Latency            : {elapsed:.3f} seconds\n")
    print(f"Current Attack Probability  : {res['current_attack_probability'] * 100.2:.1f}%")
    print(f"Current Risk Score          : {res['risk_score']:.1f} / 100")
    print(f"Risk Level Category         : {res['risk_level']}")
    print(f"Predicted MITRE Stage       : {res['stage_name']}")
    print("\n-----------------------------------------------------------------")
    print("5-STEP WORLD MODEL FUTURE RISK FORECAST:")
    print("-----------------------------------------------------------------")
    for step_i, (r, p) in enumerate(zip(res['future_risk'], res['future_attack_probability']), start=1):
        print(f"  +{(step_i*5):<2} sec: Risk {r:5.1f} / 100 | Future Attack Prob: {p*100:5.1f}%")
        
    print("\n-----------------------------------------------------------------")
    print("EARLY WARNING ASSESSMENT:")
    print("-----------------------------------------------------------------")
    print(f"  Warning Triggered        : {'YES' if res['warning_triggered'] else 'NO (Normal Baseline)'}")
    print(f"  Time to High Risk        : {res['time_to_high_risk'] if res['time_to_high_risk'] else 'N/A'}")
    print(f"  Advisory Message         : {res['warning_message']}")
    
    print("\n-----------------------------------------------------------------")
    print("EXPLAINABLE AI (XAI) TECHNICAL RATIONALE:")
    print("-----------------------------------------------------------------")
    top_feat = res['top_shap_features'][0]['feature'] if res['top_shap_features'] else "N/A"
    top_shap = res['top_shap_features'][0]['shap_value'] if res['top_shap_features'] else 0.0
    print(f"  Top SHAP Feature Driver  : {top_feat} (Contribution: {top_shap:+.4f})")
    print(f"  Most Influential Timestep: Window '{res['most_influential_timestep']}' (Weight: {res['most_influential_weight']*100:.1f}%)")
    print(f"  SHAP Technical Summary   :\n{res['shap_explanation']}")
    print(f"  Temporal Explanation     :\n{res['temporal_explanation']}")
    
    print("=" * 65)
    print("[NOTICE]: SYNTHETIC TEST FIXTURE - NOT REAL ATTACK TRAFFIC")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_sih_demo()
