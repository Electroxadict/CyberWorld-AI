"""
Environment & Dependency Verification Tool for CyberWorld-AI.
Checks Python version, required open-source libraries, PyTorch CUDA/CPU status,
and model artifact integrity.

Usage: python scripts/check_environment.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def check_env():
    print("=" * 60)
    print("      CYBERWORLD-AI ENVIRONMENT & SYSTEM INTEGRITY CHECK")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Python Version
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 9)
    print(f"[{'PASS' if py_ok else 'FAIL'}] Python Version        : {py_ver} (>= 3.9 required)")
    if not py_ok: all_passed = False
    
    # 2. Open-Source Libraries
    libraries = [
        ("torch", "PyTorch"),
        ("xgboost", "XGBoost"),
        ("shap", "SHAP"),
        ("scapy", "Scapy"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn")
    ]
    
    for mod_name, disp_name in libraries:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"[PASS] {disp_name:<21}: v{ver}")
        except ImportError as err:
            print(f"[FAIL] {disp_name:<21}: NOT INSTALLED ({err})")
            all_passed = False
            
    # 3. PyTorch CUDA / CPU Execution Device
    import torch
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU (Fallback)"
    print(f"[PASS] PyTorch Device         : {device_name}")
    
    # 4. Model Artifact Integrity
    print("-" * 60)
    print("MODEL ARTIFACT INTEGRITY CHECKS:")
    print("-" * 60)
    
    models_dir = PROJECT_ROOT / "models"
    required_artifacts = [
        "world_model.pt",
        "model_config.json",
        "scaler.pkl",
        "feature_columns.pkl",
        "xgb_risk_model.pkl",
        "xgb_stage_model.pkl",
        "xgb_feature_columns.pkl",
        "xgb_model_config.json"
    ]
    
    for art in required_artifacts:
        art_path = models_dir / art
        if art_path.exists():
            size_kb = art_path.stat().st_size / 1024.0
            print(f"[PASS] {art:<23}: {size_kb:.1f} KB")
        else:
            print(f"[FAIL] {art:<23}: MISSING")
            all_passed = False
            
    print("=" * 60)
    if all_passed:
        print("OVERALL STATUS: PASS (System fully configured & ready)")
    else:
        print("OVERALL STATUS: FAIL (Some dependencies or model artifacts missing)")
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    ok = check_env()
    sys.exit(0 if ok else 1)
