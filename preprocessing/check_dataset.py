"""
Dataset Validator and Inspector for CyberWorld-AI.
Checks data/raw/ for CIC-IDS2018 or compatible CSV datasets, inspects schema, 
maps column aliases, prints statistics, and offers synthetic dataset generation if empty.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column Alias Map for CIC-IDS2018, CTU-13, and standard network flow datasets
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "date_time", "frame.time", "start_time"],
    "label": ["label", "class", "attack", "attack_category", "target"],
    "dst_port": ["dst port", "destination port", "destination_port", "dst_port", "dport"],
    "src_port": ["src port", "source port", "source_port", "src_port", "sport"],
    "protocol": ["protocol", "proto"],
    "flow_duration": ["flow duration", "flow_duration", "duration"],
    "fwd_pkts": ["tot fwd pkts", "total fwd packets", "fwd_packets", "total_fwd_pkts"],
    "bwd_pkts": ["tot bwd pkts", "total backward packets", "bwd_packets", "total_bwd_pkts"],
    "fwd_bytes": ["totlen fwd pkts", "total length of fwd packets", "fwd_bytes", "fwd_header_len"],
    "bwd_bytes": ["totlen bwd pkts", "total length of bwd packets", "bwd_bytes", "bwd_header_len"],
}

def load_config(config_path="config.yaml"):
    """Load system configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {"paths": {"raw_data_dir": "data/raw"}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def detect_column(df_columns, alias_key):
    """Find matching column name in DataFrame using alias mapping."""
    aliases = COLUMN_ALIASES.get(alias_key, [])
    for col in df_columns:
        cleaned_col = str(col).strip().lower()
        if cleaned_col in aliases:
            return col
    return None

def generate_synthetic_dataset(output_path):
    """Generate a realistic synthetic CIC-IDS2018-style CSV file for local testing."""
    logger.info("Generating synthetic network traffic dataset for immediate local testing...")
    np.random.seed(42)
    num_samples = 2000
    
    # 5-second interval simulation over time
    base_time = pd.Timestamp("2026-08-31 10:00:00")
    timestamps = [base_time + pd.Timedelta(seconds=i * 0.5) for i in range(num_samples)]
    
    # Synthetic network features
    protocols = np.random.choice([6, 17, 1], size=num_samples, p=[0.8, 0.15, 0.05]) # TCP, UDP, ICMP
    src_ports = np.random.randint(1024, 65535, size=num_samples)
    dst_ports = np.random.choice([80, 443, 22, 53, 8080, 445], size=num_samples)
    flow_duration = np.random.exponential(scale=500000, size=num_samples)
    
    fwd_pkts = np.random.poisson(lam=10, size=num_samples) + 1
    bwd_pkts = np.random.poisson(lam=8, size=num_samples) + 1
    fwd_bytes = fwd_pkts * np.random.randint(40, 1500, size=num_samples)
    bwd_bytes = bwd_pkts * np.random.randint(40, 1500, size=num_samples)
    
    # Simulate attack progression & intermittent attacks across time
    labels = []
    syn_flags = np.random.randint(0, 2, size=num_samples)
    ack_flags = np.random.randint(0, 2, size=num_samples)
    rst_flags = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Interleave all 6 MITRE stages across the timeline
        period_idx = (i // 50) % 6
        if period_idx == 1:
            labels.append("Reconnaissance-PortScan")
            syn_flags[i] = 1
            ack_flags[i] = 0
            dst_ports[i] = int(1000 + (i % 500))
        elif period_idx == 2:
            labels.append("SSH-Bruteforce")
            dst_ports[i] = 22
            syn_flags[i] = 1
            rst_flags[i] = 1 if i % 2 == 0 else 0
        elif period_idx == 3:
            labels.append("DoS-DoS attack-HOIC")
            dst_ports[i] = 445
            fwd_pkts[i] *= 5
            fwd_bytes[i] *= 5
        elif period_idx == 4:
            labels.append("Bot-C2")
            dst_ports[i] = 8080
            fwd_pkts[i] = 2
            bwd_pkts[i] = 2
        elif period_idx == 5:
            labels.append("Data-Exfiltration")
            dst_ports[i] = 443
            fwd_bytes[i] *= 20
            bwd_bytes[i] *= 20
            rst_flags[i] = 1
        else:
            labels.append("BENIGN")

    data = {
        "Timestamp": timestamps,
        "Src Port": src_ports,
        "Dst Port": dst_ports,
        "Protocol": protocols,
        "Flow Duration": flow_duration,
        "Tot Fwd Pkts": fwd_pkts,
        "Tot Bwd Pkts": bwd_pkts,
        "TotLen Fwd Pkts": fwd_bytes,
        "TotLen Bwd Pkts": bwd_bytes,
        "Fwd Pkt Len Mean": fwd_bytes / fwd_pkts,
        "Bwd Pkt Len Mean": bwd_bytes / bwd_pkts,
        "Flow Byts/s": (fwd_bytes + bwd_bytes) / (flow_duration + 1e-5),
        "Flow Pkts/s": (fwd_pkts + bwd_pkts) / (flow_duration + 1e-5),
        "SYN Flag Cnt": syn_flags,
        "ACK Flag Cnt": ack_flags,
        "RST Flag Cnt": rst_flags,
        "Label": labels
    }
    
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset saved successfully to {output_path}")
    return output_path

def check_dataset(config_path="config.yaml"):
    """Check data/raw/ directory for dataset files and inspect schema."""
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(raw_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {raw_dir.resolve()}")
        print("\n" + "=" * 60)
        print(" DATASET NOT FOUND IN data/raw/")
        print("=" * 60)
        print("To use real CIC-IDS2018 / CTU-13 data:")
        print("1. Download CIC-IDS2018 CSV files from:")
        print("   https://www.unb.ca/cic/datasets/ids-2018.html")
        print("2. Place the CSV file(s) inside directory: data/raw/")
        print("3. Re-run this check script.")
        print("-" * 60)
        
        # Auto generate synthetic data for immediate execution
        synthetic_path = raw_dir / "synthetic_cic_ids2018.csv"
        generate_synthetic_dataset(synthetic_path)
        csv_files = [synthetic_path]
    
    print("\n" + "=" * 60)
    print(f" DATASET INSPECTION: Found {len(csv_files)} CSV File(s)")
    print("=" * 60)
    
    total_rows = 0
    for file_path in csv_files:
        print(f"\nInspecting File: {file_path.name}")
        try:
            # Read sample to inspect schema
            df_sample = pd.read_csv(file_path, nrows=5000)
            total_rows += len(df_sample)
            
            label_col = detect_column(df_sample.columns, "label")
            timestamp_col = detect_column(df_sample.columns, "timestamp")
            dst_port_col = detect_column(df_sample.columns, "dst_port")
            
            print(f"  - Total Columns Identified : {len(df_sample.columns)}")
            print(f"  - Detected Label Column   : {label_col}")
            print(f"  - Detected Timestamp Col  : {timestamp_col}")
            print(f"  - Detected Dst Port Col   : {dst_port_col}")
            
            if label_col:
                class_counts = df_sample[label_col].value_counts().to_dict()
                print("  - Class Distribution Sample:")
                for k, v in class_counts.items():
                    print(f"      * {k}: {v}")
            else:
                logger.warning(f"Could not automatically detect Label column in {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            
    print("\nDataset check complete. System ready for preprocessing.")

if __name__ == "__main__":
    check_dataset()
