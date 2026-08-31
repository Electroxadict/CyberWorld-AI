"""
Sample PCAP Generator for CyberWorld-AI.
Creates a valid synthetic .pcap file with 60 seconds of traffic (12 temporal windows)
to test and validate PCAP feature extraction and predictive defence.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np

try:
    from scapy.all import Ether, IP, TCP, UDP, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

def generate_sample_pcap(output_path="data/raw/sample_test.pcap"):
    """Generates a multi-window PCAP file using Scapy."""
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy library is required to generate PCAP test files.")
        
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    
    packets = []
    base_time = time.time() - 300.0 # 5 minutes ago
    
    np.random.seed(42)
    
    # Generate traffic across 12 temporal 5-second windows (60 seconds total)
    for window_i in range(12):
        window_start = base_time + (window_i * 5.0)
        
        # 10 to 15 packets per 5s window
        num_pkts = np.random.randint(10, 16)
        for pkt_j in range(num_pkts):
            pkt_time = window_start + (pkt_j * 0.3)
            
            src_ip = f"192.168.1.{np.random.randint(10, 50)}"
            dst_ip = "10.0.0.5"
            src_port = np.random.randint(1024, 65535)
            dst_port = int(np.random.choice([80, 443, 22, 53, 8080]))
            
            # Simulate SYN / ACK TCP packets
            flags = "S" if (window_i >= 8 and pkt_j < 4) else "A"
            
            pkt = Ether()/IP(src=src_ip, dst=dst_ip, ttl=64)/TCP(sport=src_port, dport=dst_port, flags=flags)
            pkt.time = pkt_time
            packets.append(pkt)
            
    wrpcap(str(output_p), packets)
    print(f"Sample PCAP generated with {len(packets)} packets across 12 temporal windows: {output_p.resolve()}")
    return output_p

if __name__ == "__main__":
    generate_sample_pcap()
