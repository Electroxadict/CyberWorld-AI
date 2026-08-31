"""
Master CLI Pipeline Entry Point for CyberWorld-AI.
Executes end-to-end predictive cyber-defence pipeline:
PCAP -> Scapy -> 5s Windows -> Scaler -> 10-step Seq -> World Model -> 5-step Rollout -> XGBoost -> Risk Engine -> Early Warning -> SHAP -> Temporal Attention -> Report.

Usage:
  python scripts/run_pipeline.py --pcap data/raw/sample_test.pcap --horizon 5 --explain --json --verbose
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import time
from datetime import datetime

from inference.pcap_pipeline import PCAPPredictivePipeline
from scripts.generate_sample_pcap import generate_sample_pcap

def main():
    parser = argparse.ArgumentParser(description="CyberWorld-AI Master Predictive Defence Pipeline CLI")
    parser.add_argument("--pcap", type=str, default=None, help="Path to raw .pcap or .pcapng file")
    parser.add_argument("--horizon", type=int, default=5, help="Future rollout forecast horizon steps (default: 5)")
    parser.add_argument("--explain", action="store_true", help="Include detailed SHAP and Temporal Attention explanations")
    parser.add_argument("--json", action="store_true", help="Export structured JSON report to logs/latest_analysis.json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    pcap_path = args.pcap
    if pcap_path is None:
        default_pcap = PROJECT_ROOT / "data" / "raw" / "sample_test.pcap"
        if not default_pcap.exists():
            logger.info("No PCAP path provided. Generating sample test PCAP...")
            generate_sample_pcap(default_pcap)
        pcap_path = str(default_pcap)
        
    pcap_p = Path(pcap_path).resolve()
    if not pcap_p.exists():
        print(f"Error: Specified PCAP file not found: {pcap_p}")
        sys.exit(1)
        
    is_synthetic = pcap_p.name == "sample_test.pcap"
    
    try:
        pipeline = PCAPPredictivePipeline()
        start_time = time.time()
        res = pipeline.predict(pcap_p)
        elapsed_sec = time.time() - start_time
        
        # --- Formatted Master CLI Output ---
        print("\n" + "=" * 50)
        print("        CYBERWORLD-AI")
        print("   PREDICTIVE CYBER DEFENCE")
        print("=" * 50)
        print(f"PCAP:\n{res['pcap_file']}")
        print(f"\nFlows:\n{res['flow_count']}")
        print(f"\nTemporal Windows:\n{res['window_count']}")
        print("\nSequence:\n10 x 69")
        print("\n" + "-" * 50)
        print("CURRENT STATE")
        print("-" * 50)
        print(f"Attack Probability:\n{res['current_attack_probability'] * 100:.2f}%")
        print(f"\nRisk Score:\n{res['risk_score']:.2f} / 100")
        print(f"\nRisk Level:\n{res['risk_level']}")
        print(f"\nPredicted Attack Stage:\n{res['stage_name']}")
        print("\n" + "-" * 50)
        print("FUTURE FORECAST")
        print("-" * 50)
        for step_i, (r, p) in enumerate(zip(res['future_risk'], res['future_attack_probability']), start=1):
            sec = step_i * 5
            print(f"+{sec:<2} sec:\nRisk {r:.2f} | Attack Probability {p*100:.2f}%\n")
            
        print("-" * 50)
        print("EARLY WARNING")
        print("-" * 50)
        trig_str = "YES" if res['warning_triggered'] else "NO"
        print(f"Triggered:\n{trig_str}")
        t_high = res['time_to_high_risk']
        if t_high is not None:
            print(f"\nTime to High Risk:\napproximately {t_high} seconds")
        else:
            print("\nTime to High Risk:\nN/A (Low Risk)")
        print(f"\nMessage:\n{res['warning_message']}")
        
        if args.explain:
            print("\n" + "-" * 50)
            print("EXPLAINABILITY")
            print("-" * 50)
            top_feat = res['top_shap_features'][0]['feature'] if res['top_shap_features'] else "N/A"
            top_shap_val = res['top_shap_features'][0]['shap_value'] if res['top_shap_features'] else 0.0
            print(f"Top SHAP Feature:\n{top_feat} ({top_shap_val:+.4f})")
            print(f"\nMost Influential Timestep:\n{res['most_influential_timestep']}")
            print(f"\nExplanation:\n{res['shap_explanation']}")
            print(f"\nTemporal Explanation:\n{res['temporal_explanation']}")
            
        print("=" * 50)
        if is_synthetic:
            print("[NOTICE]: SYNTHETIC TEST FIXTURE - NOT REAL ATTACK TRAFFIC")
        print("=" * 50 + "\n")
        
        # --- Save JSON if requested ---
        if args.json:
            logs_dir = PROJECT_ROOT / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            json_path = logs_dir / "latest_analysis.json"
            
            json_export = {
                "timestamp": datetime.now().isoformat(),
                "pcap_file": res["pcap_file"],
                "packet_count": res["flow_count"],
                "flow_count": res["flow_count"],
                "temporal_window_count": res["window_count"],
                "sequence_shape": [1, 10, 69],
                "current_attack_probability": res["current_attack_probability"],
                "current_risk": res["risk_score"],
                "risk_level": res["risk_level"],
                "future_attack_probabilities": res["future_attack_probability"],
                "future_risks": res["future_risk"],
                "predicted_stage": res["predicted_stage"],
                "stage_name": res["stage_name"],
                "stage_probabilities": res["stage_probabilities"],
                "warning_triggered": res["warning_triggered"],
                "time_to_high_risk": res["time_to_high_risk"],
                "warning_message": res["warning_message"],
                "top_shap_features": res["top_shap_features"],
                "feature_group_importance": res.get("feature_group_importance", {}),
                "attention_weights": res["attention_weights"],
                "most_influential_timestep": res["most_influential_timestep"],
                "temporal_explanation": res["temporal_explanation"],
                "pipeline_status": "COMPLETE",
                "execution_time_seconds": round(elapsed_sec, 3),
                "is_synthetic": is_synthetic
            }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_export, f, indent=2)
            logger.info(f"Saved analysis JSON report to {json_path}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
