"""
Attack Mapping package for CyberWorld-AI.
Maps network behaviors and attack labels to transparent MITRE ATT&CK stages.
"""

from attack_mapping.mitre_mapper import (
    MitreStageMapper,
    map_label_to_stage,
    map_behaviour_to_stage,
    get_stage_name,
    get_stage_description,
    explain_mapping
)

__all__ = [
    "MitreStageMapper",
    "map_label_to_stage",
    "map_behaviour_to_stage",
    "get_stage_name",
    "get_stage_description",
    "explain_mapping"
]
