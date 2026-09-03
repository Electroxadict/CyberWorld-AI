"""
SIH Demo PCAP Scenario Generator for CyberWorld-AI.
Generates 5 distinct network traffic PCAP scenario files for live presentation:
1. PCAP 1 — Normal Traffic (Benign web/DNS traffic)
2. PCAP 2 — Port Scanning / Reconnaissance (Multi-port SYN scanning)
3. PCAP 3 — Brute Force / Initial Access (SSH/FTP high-rate authentication attempts)
4. PCAP 4 — Lateral Movement (Internal subnet SMB/RDP traffic)
5. PCAP 5 — Data Exfiltration (Outbound heavy data burst)

Usage: python scripts/generate_sih_demo_pcaps.py
"""

import time
from pathlib import Path

try:
    from scapy.all import Ether, IP, TCP, UDP, wrpcap
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_all_sih_pcaps(output_dir=None):
    if not SCAPY_OK:
        print("[ERROR] Scapy is required to generate PCAP scenario files. Install via `pip install scapy`.")
        return False
        
    if output_dir is None:
        out_dir = PROJECT_ROOT / "data" / "raw"
    else:
        out_dir = Path(output_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    base_t = time.time() - 600.0

    print("Generating 5 SIH Demo PCAP Scenarios in data/raw/...")

    # 1. Normal Traffic
    pkts1 = []
    for w in range(12):
        t_w = base_t + w * 5.0
        for i in range(5):
            p = Ether()/IP(src=f"192.168.1.{10+i}", dst="10.0.0.5")/TCP(sport=10000+w*10+i, dport=443, flags="A")
            p.time = t_w + i * 0.5
            pkts1.append(p)
    f1 = out_dir / "pcap_1_normal.pcap"
    wrpcap(str(f1), pkts1)
    print(f"  [+] Created {f1.name}: {len(pkts1)} packets across 12 temporal windows.")

    # 2. Port Scanning / Reconnaissance
    pkts2 = []
    for w in range(12):
        t_w = base_t + w * 5.0
        for i in range(15):
            p = Ether()/IP(src="192.168.1.99", dst="10.0.0.5")/TCP(sport=20000+w*20+i, dport=1+w*100+i*5, flags="S")
            p.time = t_w + i * 0.2
            pkts2.append(p)
    f2 = out_dir / "pcap_2_reconnaissance.pcap"
    wrpcap(str(f2), pkts2)
    print(f"  [+] Created {f2.name}: {len(pkts2)} packets across 12 temporal windows.")

    # 3. Brute Force / Initial Access
    pkts3 = []
    for w in range(12):
        t_w = base_t + w * 5.0
        for i in range(20):
            flags = "S" if i % 2 == 0 else "R"
            p = Ether()/IP(src="192.168.1.88", dst="10.0.0.5")/TCP(sport=30000+w*30+i, dport=22, flags=flags)
            p.time = t_w + i * 0.15
            pkts3.append(p)
    f3 = out_dir / "pcap_3_initial_access.pcap"
    wrpcap(str(f3), pkts3)
    print(f"  [+] Created {f3.name}: {len(pkts3)} packets across 12 temporal windows.")

    # 4. Lateral Movement
    pkts4 = []
    for w in range(12):
        t_w = base_t + w * 5.0
        for i in range(15):
            src_ip = f"10.0.0.{10 + (i % 4)}"
            dst_ip = f"10.0.0.{20 + (i % 4)}"
            p = Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=40000+w*20+i, dport=445, flags="PA")
            p.time = t_w + i * 0.25
            pkts4.append(p)
    f4 = out_dir / "pcap_4_lateral_movement.pcap"
    wrpcap(str(f4), pkts4)
    print(f"  [+] Created {f4.name}: {len(pkts4)} packets across 12 temporal windows.")

    # 5. Data Exfiltration
    pkts5 = []
    for w in range(12):
        t_w = base_t + w * 5.0
        for i in range(25):
            payload = b"E" * 1400
            p = Ether()/IP(src="10.0.0.5", dst="198.51.100.44")/TCP(sport=50000+w*10+i, dport=443, flags="PA")/payload
            p.time = t_w + i * 0.1
            pkts5.append(p)
    f5 = out_dir / "pcap_5_exfiltration.pcap"
    wrpcap(str(f5), pkts5)
    print(f"  [+] Created {f5.name}: {len(pkts5)} packets across 12 temporal windows.")

    print("\n[SUCCESS] All 5 SIH Demo PCAPs ready in data/raw/.\n")
    return True

if __name__ == "__main__":
    generate_all_sih_pcaps()
