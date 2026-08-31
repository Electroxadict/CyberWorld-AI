"""
Unit tests for MITRE ATT&CK Stage Mapper.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attack_mapping.mitre_mapper import (
    MitreStageMapper,
    map_label_to_stage,
    map_behaviour_to_stage,
    get_stage_name,
    get_stage_description,
    explain_mapping
)

def test_label_mapping():
    # 1. Normal
    res_benign = map_label_to_stage("BENIGN")
    assert res_benign["stage_id"] == 0
    assert res_benign["stage_name"] == "Normal"

    # 2. Reconnaissance
    res_recon = map_label_to_stage("Reconnaissance-PortScan")
    assert res_recon["stage_id"] == 1
    assert res_recon["stage_name"] == "Reconnaissance"

    # 3. Initial Access
    res_ssh = map_label_to_stage("SSH-Bruteforce")
    assert res_ssh["stage_id"] == 2
    assert res_ssh["stage_name"] == "Initial Access"

    # 4. Unknown label handling
    res_unknown = map_label_to_stage("Custom-ZeroDay-Exploit")
    assert res_unknown["stage_id"] == 1
    assert res_unknown["stage_name"] == "Unclassified Attack"
    assert res_unknown["is_classified"] is False

def test_behavioral_mapping():
    high_scan_dict = {
        "Port_Scan_Score": 0.8,
        "SYN_ACK_Ratio": 3.0,
        "Port_Diversity": 0.5
    }
    res_scan = map_behaviour_to_stage(high_scan_dict)
    assert res_scan["stage_id"] == 1
    assert "Reconnaissance" in res_scan["stage_name"]

def test_stage_explanations():
    exp = explain_mapping(2)
    assert exp["stage_id"] == 2
    assert "Initial Access" in exp["stage_name"]
    assert "description" in exp
