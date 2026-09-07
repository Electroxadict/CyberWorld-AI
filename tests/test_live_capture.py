"""
Unit and Integration tests for CyberWorld-AI Real-Time Network Monitoring & Live Capture Engine.
Tests interface discovery, canonical flow hashing, packet header parsing, 5s window feature
compilation (69 features), rolling telemetry buffers, and start/stop controls.
"""

import sys
from pathlib import Path
import time
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.live_capture import LiveCaptureEngine, get_available_interfaces
from inference.pcap_pipeline import PCAPPredictivePipeline

def test_interface_discovery():
    """Verifies interface detection returns non-empty list with Wi-Fi / Ethernet / Loopback."""
    ifaces = get_available_interfaces()
    assert isinstance(ifaces, list)
    assert len(ifaces) > 0
    # Must contain standard user interfaces
    has_standard = any("wi-fi" in i.lower() or "ethernet" in i.lower() or "loopback" in i.lower() for i in ifaces)
    assert has_standard is True

def test_canonical_flow_key():
    """Verifies bidirectional flows map to the same canonical key."""
    engine = LiveCaptureEngine()
    
    k1 = engine.canonical_flow_key("192.168.1.10", "8.8.8.8", 54321, 53, 17)
    k2 = engine.canonical_flow_key("8.8.8.8", "192.168.1.10", 53, 54321, 17)
    
    # 5-tuple flow key must match
    assert k1[:5] == k2[:5]
    # Direction flag must be inverted
    assert k1[5] != k2[5]

def test_window_feature_compilation_schema():
    """Verifies 5-second window aggregator outputs exact 69 training feature columns."""
    engine = LiveCaptureEngine()
    
    # Test quiet / baseline compilation
    feats = engine._compile_window_features({})
    assert isinstance(feats, dict)
    assert "Flow_Count" in feats
    assert "Unique_Src_Ports" in feats
    assert "Unique_Dst_Ports" in feats
    assert "SYN_ACK_Ratio" in feats
    assert "Port_Diversity" in feats
    assert "Connection_Failure_Rate" in feats

    # Verify matching with persisted 69 feature schema
    pipeline = PCAPPredictivePipeline()
    expected_cols = pipeline.feature_columns
    assert len(expected_cols) == 69
    
    for col in expected_cols:
        assert col in feats, f"Missing expected 69-feature column: {col}"

def test_live_engine_lifecycle_and_telemetry():
    """Verifies LiveCaptureEngine starts, streams packets, populates tables, and stops cleanly."""
    engine = LiveCaptureEngine()
    engine.start(interface="Wi-Fi")
    assert engine.is_running is True
    
    # Let capture run for 1.5 seconds
    time.sleep(1.5)
    
    metrics = engine.get_live_metrics()
    assert metrics["status"] == "LIVE"
    assert metrics["interface"] == "Wi-Fi"
    assert metrics["total_packets"] > 0
    assert metrics["total_flows"] > 0
    
    conn_table = engine.get_connection_table()
    assert isinstance(conn_table, pd.DataFrame)
    assert len(conn_table) > 0
    assert "Source IP" in conn_table.columns
    assert "Destination IP" in conn_table.columns
    assert "Status" in conn_table.columns
    
    proto_dist = engine.get_protocol_distribution()
    assert "TCP" in proto_dist
    assert "UDP" in proto_dist
    assert "ICMP" in proto_dist
    assert "ARP" in proto_dist
    assert proto_dist["TCP"]["count"] > 0
    
    top_src, top_dst = engine.get_top_talkers()
    assert len(top_src) > 0
    assert len(top_dst) > 0
    
    topo = engine.get_topology_data()
    assert "nodes" in topo
    assert "links" in topo
    assert len(topo["nodes"]) >= 2
    
    # Stop capture
    engine.stop()
    assert engine.is_running is False
    assert engine.get_live_metrics()["status"] == "STOPPED"

def test_live_pipeline_prediction():
    """Verifies that windows generated from LiveCaptureEngine feed into PCAPPredictivePipeline."""
    engine = LiveCaptureEngine()
    engine.start(interface="Wi-Fi")
    time.sleep(1.0)
    engine.stop()
    
    # Get sequence dataframe (padded to 10 states)
    df_seq = engine.get_latest_sequence_dataframe()
    assert len(df_seq) == 10
    
    pipeline = PCAPPredictivePipeline()
    res = pipeline.predict_windows(
        df_windows=df_seq,
        source_name="Live Monitoring Test",
        identifier="Wi-Fi",
        flow_count=100
    )
    
    assert res["source"] == "Live Monitoring Test"
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 100.0
    assert "risk_level" in res
    assert len(res["future_risk"]) == 5
    assert len(res["future_attack_probability"]) == 5
    assert "stage_name" in res
    assert len(res["top_shap_features"]) > 0
    assert len(res["attention_weights"]) == 10
