"""
MITRE ATT&CK Stage Mapper for CyberWorld-AI.
Provides transparent engineering mapping from network traffic behaviors and dataset labels
to six key attack progression stages: Normal, Reconnaissance, Initial Access, Lateral Movement, C2, Exfiltration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAGE_DESCRIPTIONS = {
    0: ("Normal", "Benign, authorized background network activity within baseline parameters."),
    1: ("Reconnaissance", "Adversary gathering host, port, or network service information (e.g., port scanning)."),
    2: ("Initial Access", "Adversary attempting entry into the network (e.g., brute-force, web vulnerability exploitation)."),
    3: ("Lateral Movement", "Adversary expanding access across internal systems and network segments."),
    4: ("Command and Control", "Adversary maintaining beaconing, remote control sessions, or botnet communications."),
    5: ("Exfiltration", "Adversary stealing data or transferring large encrypted volumes out of the perimeter.")
}

# Explicit Rule-based Keyword Mapping Matrix
LABEL_RULE_MATRIX = [
    # Normal
    (["benign", "normal", "0"], 0, "Normal", "Traffic matches benign authorized baseline activity."),
    
    # Reconnaissance (Stage 1)
    (["portscan", "reconnaissance", "nmap", "ipsweep", "satan", "mscan"], 1, "Reconnaissance", "Port scanning or active host discovery activity detected."),
    
    # Initial Access (Stage 2)
    (["ssh-bruteforce", "ftp-bruteforce", "brute force", "patator", "web attack", "sqli", "xss"], 2, "Initial Access", "Brute-force authentication or web application exploitation attempt detected."),
    
    # Lateral Movement / DoS (Stage 3)
    (["dos", "ddos", "hoic", "loic", "goldeneye", "hulk", "slowloris", "smurf"], 3, "Lateral Movement / DoS", "High-volume resource exhaustion or internal denial-of-service traffic detected."),
    
    # Command and Control (Stage 4)
    (["bot", "c2", "command and control", "infilteration", "beacon"], 4, "Command and Control", "Periodic suspicious outbound C2 beaconing or internal botnet traffic detected."),
    
    # Exfiltration (Stage 5)
    (["exfiltration", "data-exfiltration", "data_leak"], 5, "Exfiltration", "Abnormally large or anomalous outbound data transfer detected.")
]

class MitreStageMapper:
    """Engine mapping network labels and behavioral feature vectors to MITRE ATT&CK stages."""
    
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.stage_names = self.config.get("mitre_stages", {i: desc[0] for i, desc in STAGE_DESCRIPTIONS.items()})

    def map_label_to_stage(self, label_str: str) -> dict:
        """
        Maps a dataset string attack label to MITRE stage index, stage name, and explanation.
        
        Args:
            label_str (str): Raw dataset label (e.g., "Reconnaissance-PortScan").
            
        Returns:
            dict: {stage_id, stage_name, description, reason, is_classified}
        """
        if not isinstance(label_str, str) or not label_str.strip():
            return {
                "stage_id": 0,
                "stage_name": "Normal",
                "description": STAGE_DESCRIPTIONS[0][1],
                "reason": "Missing or non-string label defaulted to Normal.",
                "is_classified": False
            }
            
        clean_label = label_str.strip().lower()
        
        for keywords, stage_id, stage_name, reason in LABEL_RULE_MATRIX:
            for kw in keywords:
                if kw in clean_label:
                    return {
                        "stage_id": stage_id,
                        "stage_name": self.stage_names.get(stage_id, stage_name),
                        "description": STAGE_DESCRIPTIONS.get(stage_id, ("Unknown", ""))[1],
                        "reason": f"Label '{label_str}' matched rule: {reason}",
                        "is_classified": True
                    }
                    
        # Explicit fallback for unknown attack labels
        return {
            "stage_id": 1,
            "stage_name": "Unclassified Attack",
            "description": "Anomalous traffic label not explicitly in rule matrix.",
            "reason": f"Unrecognized attack label '{label_str}'. Assigned to fallback stage 'Unclassified Attack'.",
            "is_classified": False
        }

    def map_behaviour_to_stage(self, feature_dict: dict) -> dict:
        """
        Infers MITRE stage directly from macro network traffic behavioral metrics.
        
        Args:
            feature_dict (dict): Dictionary of behavioral features (e.g., Port_Scan_Score, SYN_ACK_Ratio).
            
        Returns:
            dict: {stage_id, stage_name, description, reason}
        """
        port_scan_score = feature_dict.get("Port_Scan_Score", 0.0)
        syn_ack_ratio = feature_dict.get("SYN_ACK_Ratio", 0.0)
        rst_syn_ratio = feature_dict.get("RST_SYN_Ratio", 0.0)
        burst_score = feature_dict.get("Traffic_Burst_Score", 0.0)
        port_div = feature_dict.get("Port_Diversity", 0.0)
        
        if port_scan_score > 0.5 or (port_div > 0.4 and syn_ack_ratio > 2.0):
            return {
                "stage_id": 1,
                "stage_name": "Reconnaissance",
                "description": STAGE_DESCRIPTIONS[1][1],
                "reason": f"High port diversity ({port_div:.2f}) and port scan score ({port_scan_score:.2f}) indicate active Reconnaissance."
            }
            
        if burst_score > 5.0 and rst_syn_ratio > 0.3:
            return {
                "stage_id": 3,
                "stage_name": "Lateral Movement / DoS",
                "description": STAGE_DESCRIPTIONS[3][1],
                "reason": f"High traffic burstiness score ({burst_score:.1f}) and RST ratio indicate connection flooding."
            }
            
        return {
            "stage_id": 0,
            "stage_name": "Normal",
            "description": STAGE_DESCRIPTIONS[0][1],
            "reason": "Behavioral indicators remain within standard operational bounds."
        }

    def get_stage_name(self, stage_id: int) -> str:
        """Returns string name for numerical stage index."""
        return self.stage_names.get(int(stage_id), f"Stage-{stage_id}")

    def get_stage_description(self, stage_id: int) -> str:
        """Returns detailed description for numerical stage index."""
        return STAGE_DESCRIPTIONS.get(int(stage_id), ("Unknown", "No description available."))[1]

    def explain_mapping(self, stage_id_or_label) -> dict:
        """Provides human-readable technical rationale for stage or label."""
        if isinstance(stage_id_or_label, (int, float, np.integer)):
            sid = int(stage_id_or_label)
            return {
                "stage_id": sid,
                "stage_name": self.get_stage_name(sid),
                "description": self.get_stage_description(sid),
                "reason": f"Mapped directly from numeric stage index {sid}."
            }
        else:
            return self.map_label_to_stage(str(stage_id_or_label))

# Standalone helper functions
_default_mapper = MitreStageMapper()

def map_label_to_stage(label_str: str) -> dict:
    return _default_mapper.map_label_to_stage(label_str)

def map_behaviour_to_stage(feature_dict: dict) -> dict:
    return _default_mapper.map_behaviour_to_stage(feature_dict)

def get_stage_name(stage_id: int) -> str:
    return _default_mapper.get_stage_name(stage_id)

def get_stage_description(stage_id: int) -> str:
    return _default_mapper.get_stage_description(stage_id)

def explain_mapping(stage_id_or_label) -> dict:
    return _default_mapper.explain_mapping(stage_id_or_label)

if __name__ == "__main__":
    print("MITRE Stage Mapper Test:")
    print("1. PortScan Mapping   :", map_label_to_stage("Reconnaissance-PortScan"))
    print("2. SSH Bruteforce     :", map_label_to_stage("SSH-Bruteforce"))
    print("3. Unknown Label      :", map_label_to_stage("Unknown-ZeroDay-Vector"))
