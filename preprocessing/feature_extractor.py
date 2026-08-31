"""
Offline PCAP / SCAPY Feature Extractor for CyberWorld-AI.
Parses raw .pcap and .pcapng files using Scapy, associates bidirectional canonical 5-tuple flows,
handles missing network layers safely, and outputs flow tables matching the training schema.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np

try:
    from scapy.all import rdpcap, PcapReader, IP, IPv6, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from preprocessing.check_dataset import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PCAPFeatureExtractor:
    """Production-grade Scapy PCAP feature extraction engine."""
    
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.flows = {}
        self.prev_timestamp = None

    def canonical_flow_key(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: int) -> tuple:
        """
        Creates a canonical bidirectional 5-tuple flow key.
        Maps forward (src -> dst) and reverse (dst -> src) traffic to the same flow tuple.
        """
        ep1 = (src_ip, src_port)
        ep2 = (dst_ip, dst_port)
        if ep1 <= ep2:
            return (src_ip, dst_ip, src_port, dst_port, proto, True)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto, False)

    def extract_packet(self, pkt):
        """Safely parses packet layers without crashing on malformed/non-IP frames."""
        try:
            timestamp = float(pkt.time)
            length = len(pkt)
            
            # 1. IP Layer Inspection
            if pkt.haslayer(IP):
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = int(pkt[IP].proto)
                ttl = int(pkt[IP].ttl)
            elif pkt.haslayer(IPv6):
                src_ip = pkt[IPv6].src
                dst_ip = pkt[IPv6].dst
                proto = int(pkt[IPv6].nh)
                ttl = int(pkt[IPv6].hlim)
            else:
                # Non-IP frames (ARP, Ethernet L2)
                src_ip = "0.0.0.0"
                dst_ip = "0.0.0.0"
                proto = 0
                ttl = 64

            # 2. Transport Layer Inspection
            src_port = 0
            dst_port = 0
            syn_flag = 0
            ack_flag = 0
            rst_flag = 0
            fin_flag = 0
            psh_flag = 0
            urg_flag = 0
            tcp_win = 0
            
            if pkt.haslayer(TCP):
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
                flags = str(pkt[TCP].flags)
                syn_flag = 1 if "S" in flags else 0
                ack_flag = 1 if "A" in flags else 0
                rst_flag = 1 if "R" in flags else 0
                fin_flag = 1 if "F" in flags else 0
                psh_flag = 1 if "P" in flags else 0
                urg_flag = 1 if "U" in flags else 0
                tcp_win = int(pkt[TCP].window)
            elif pkt.haslayer(UDP):
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)
                
            iat = (timestamp - self.prev_timestamp) if self.prev_timestamp is not None else 0.0
            self.prev_timestamp = timestamp
            
            # Build canonical bidirectional flow key
            canonical_key = self.canonical_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
            flow_key = canonical_key[:5]
            is_forward = canonical_key[5]
            
            if flow_key not in self.flows:
                self.flows[flow_key] = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "timestamps": [timestamp],
                    "fwd_lengths": [length] if is_forward else [],
                    "bwd_lengths": [] if is_forward else [length],
                    "ttls": [ttl],
                    "tcp_wins": [tcp_win],
                    "iats": [iat],
                    "syn_cnt": syn_flag,
                    "ack_cnt": ack_flag,
                    "rst_cnt": rst_flag,
                    "fin_cnt": fin_flag,
                    "psh_cnt": psh_flag,
                    "urg_cnt": urg_flag,
                    "fwd_pkts": 1 if is_forward else 0,
                    "bwd_pkts": 0 if is_forward else 1,
                    "fwd_bytes": length if is_forward else 0,
                    "bwd_bytes": 0 if is_forward else length
                }
            else:
                f = self.flows[flow_key]
                f["end_time"] = timestamp
                f["timestamps"].append(timestamp)
                if is_forward:
                    f["fwd_lengths"].append(length)
                    f["fwd_pkts"] += 1
                    f["fwd_bytes"] += length
                else:
                    f["bwd_lengths"].append(length)
                    f["bwd_pkts"] += 1
                    f["bwd_bytes"] += length
                    
                f["ttls"].append(ttl)
                f["tcp_wins"].append(tcp_win)
                f["iats"].append(iat)
                f["syn_cnt"] += syn_flag
                f["ack_cnt"] += ack_flag
                f["rst_cnt"] += rst_flag
                f["fin_cnt"] += fin_flag
                f["psh_cnt"] += psh_flag
                f["urg_cnt"] += urg_flag
                
        except Exception:
            # Handle malformed/corrupt packet gracefully
            pass

    def extract(self, pcap_path, max_packets=None) -> pd.DataFrame:
        """
        Parses PCAP file and returns pandas DataFrame of flow records.
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy library is required for PCAP feature extraction. Install via `pip install scapy`.")
            
        pcap_p = Path(pcap_path)
        if not pcap_p.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_p.resolve()}")
            
        ext = pcap_p.suffix.lower()
        if ext not in [".pcap", ".pcapng"]:
            raise ValueError(f"Unsupported file extension '{ext}'. Expected .pcap or .pcapng file.")
            
        logger.info(f"Reading PCAP file: {pcap_p.name}...")
        self.flows = {}
        self.prev_timestamp = None
        
        count = 0
        try:
            # Use PcapReader for memory-efficient streaming
            with PcapReader(str(pcap_p)) as pcap_reader:
                for pkt in pcap_reader:
                    self.extract_packet(pkt)
                    count += 1
                    if max_packets is not None and count >= max_packets:
                        break
        except Exception as e:
            # Fallback to rdpcap if PcapReader fails
            logger.warning(f"Streaming PcapReader failed ({e}). Retrying with rdpcap...")
            packets = rdpcap(str(pcap_p))
            for pkt in packets:
                self.extract_packet(pkt)
                count += 1
                if max_packets is not None and count >= max_packets:
                    break
                    
        if count == 0:
            raise ValueError(f"PCAP file {pcap_p.name} is empty or contains no readable packets.")
            
        logger.info(f"Parsed {count} packets into {len(self.flows)} canonical flows.")
        return self.build_flow_records()

    def build_flow_records(self) -> pd.DataFrame:
        """Converts internal flows dict into pandas DataFrame matching CSV preprocessed schema."""
        records = []
        for (src_ip, dst_ip, src_port, dst_port, proto), f in self.flows.items():
            duration_us = max((f["end_time"] - f["start_time"]) * 1000000.0, 1.0)
            duration_sec = duration_us / 1000000.0
            
            fwd_pkts = f["fwd_pkts"]
            bwd_pkts = f["bwd_pkts"]
            tot_pkts = fwd_pkts + bwd_pkts
            
            fwd_bytes = f["fwd_bytes"]
            bwd_bytes = f["bwd_bytes"]
            tot_bytes = fwd_bytes + bwd_bytes
            
            fwd_len_mean = np.mean(f["fwd_lengths"]) if f["fwd_lengths"] else 0.0
            bwd_len_mean = np.mean(f["bwd_lengths"]) if f["bwd_lengths"] else 0.0
            
            rec = {
                "Timestamp": pd.to_datetime(f["start_time"], unit="s"),
                "Src Port": src_port,
                "Dst Port": dst_port,
                "Protocol": proto,
                "Flow Duration": duration_us,
                "Tot Fwd Pkts": fwd_pkts,
                "Tot Bwd Pkts": bwd_pkts,
                "TotLen Fwd Pkts": fwd_bytes,
                "TotLen Bwd Pkts": bwd_bytes,
                "Fwd Pkt Len Mean": fwd_len_mean,
                "Bwd Pkt Len Mean": bwd_len_mean,
                "Flow Byts/s": float(tot_bytes / (duration_sec + 1e-5)),
                "Flow Pkts/s": float(tot_pkts / (duration_sec + 1e-5)),
                "SYN Flag Cnt": f["syn_cnt"],
                "ACK Flag Cnt": f["ack_cnt"],
                "RST Flag Cnt": f["rst_cnt"],
                "TTL Mean": float(np.mean(f["ttls"])) if f["ttls"] else 64.0,
                "TCP Win Mean": float(np.mean(f["tcp_wins"])) if f["tcp_wins"] else 0.0,
                "Label": "BENIGN"
            }
            records.append(rec)
            
        df_pcap = pd.DataFrame(records)
        if "Timestamp" in df_pcap.columns:
            df_pcap = df_pcap.sort_values("Timestamp").reset_index(drop=True)
        return df_pcap

# Legacy convenience wrapper for backward compatibility
def extract_pcap_features(pcap_file_path, output_csv_path=None, config_path="config.yaml"):
    extractor = PCAPFeatureExtractor(config_path=config_path)
    df_flows = extractor.extract(pcap_file_path)
    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_flows.to_csv(out_p, index=False)
        logger.info(f"Saved extracted PCAP flows to {out_p}")
    return df_flows

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_pcap_features(sys.argv[1])
    else:
        logger.info("PCAPFeatureExtractor ready.")
