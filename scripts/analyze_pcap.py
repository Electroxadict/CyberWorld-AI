"""
PCAP Analysis CLI Tool for CyberWorld-AI.
Usage: python scripts/analyze_pcap.py path/to/file.pcap
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from inference.pcap_pipeline import PCAPPredictivePipeline
from scripts.generate_sample_pcap import generate_sample_pcap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CyberWorld-AI PCAP Analysis CLI Tool")
    parser.add_argument("pcap_path", nargs="?", default=None, help="Path to input .pcap or .pcapng file")
    args = parser.parse_args()
    
    pcap_path = args.pcap_path
    if pcap_path is None:
        pcap_path = PROJECT_ROOT / "data" / "raw" / "sample_test.pcap"
        if not pcap_path.exists():
            print("No PCAP path provided. Generating sample PCAP for test run...")
            generate_sample_pcap(pcap_path)
            
    pcap_p = Path(pcap_path)
    if not pcap_p.exists():
        print(f"Error: Specified PCAP file not found: {pcap_p}")
        sys.exit(1)
        
    try:
        pipeline = PCAPPredictivePipeline()
        res = pipeline.predict(pcap_p)
        
        print("\n" + "=" * 50)
        print("CyberWorld-AI PCAP Analysis")
        print("=" * 50)
        print(f"PCAP:\n{res['pcap_file']}\n")
        print(f"Flows:\n{res['flow_count']}\n")
        print(f"Temporal Windows:\n{res['window_count']}\n")
        print("Sequence:\n10 x 69\n")
        print(f"Current Attack Probability:\n{res['current_attack_probability'] * 100:.2f}%\n")
        print(f"Current Risk:\n{res['risk_score']:.2f} / 100\n")
        print(f"Risk Level:\n{res['risk_level']}\n")
        print(f"Predicted Stage:\n{res['stage_name']}\n")
        print("Future Risk:\n")
        for k, (r, p) in enumerate(zip(res['future_risk'], res['future_attack_probability']), start=1):
            print(f"+{(k*5):<2} sec   {r:5.1f} / 100  (Attack Prob: {p*100:5.1f}%)")
        print("")
        
        t_high = res['time_to_high_risk']
        if t_high is not None:
            print(f"Time to High Risk:\napproximately {t_high} seconds\n")
        else:
            print("Time to High Risk:\nN/A (Low Risk)\n")
            
        top_shap = res['top_shap_features'][0]['feature'] if res['top_shap_features'] else "N/A"
        print(f"Top SHAP Feature:\n{top_shap}\n")
        print(f"Most Influential Timestep:\n{res['most_influential_timestep']}\n")
        print(f"Warning:\n{res['warning_message']}\n")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to complete PCAP analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
