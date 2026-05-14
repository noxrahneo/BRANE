#!/usr/bin/env python3
"""00_setup_environment.py: sanity check for the BRANE project."""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REQUIRED_CONFIG_KEYS = ("paths", "qc", "normalization", "network")
REQUIRED_PACKAGES = ("numpy", "pandas", "scipy", "scanpy", "anndata", "sklearn", "matplotlib", "seaborn", "networkx")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for the BRANE project.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode: exit on first error.")
    return parser.parse_args()

def main() -> int: 
    args = parse_args()
    repo_root = Path (__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    
    issues: list[str] = []
    
    #load config
    if not config_path.exists():
        issues.append(f"Config file not found at {config_path}")
        config = {} 
    else:
        with config_path.open ("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
    #basic config keys
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            issues.append(f"Missing required config key: '{key}'")
            
    #resolve and check paths
    paths = config.get("paths", {})
    resolved = {k: (repo_root / v).resolve() for k, v in paths.items()}
    if "data_dir" in resolved and not resolved["data_dir"].exists():
        issues.append(f"Data directory not found at {resolved['data_dir']}")
        
    for key in ("results_dir", "logs_dir"):
        if key in resolved:
            resolved[key].mkdir(parents=True, exist_ok=True)
            
    #check packages
    versions = {} 
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            missing.append(pkg)
            
    if missing: 
        issues.append(f"Missing required packages: {', '.join(missing)}")
    
    #report results
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "config_path": str(config_path),
        "paths": {k: str(v) for k, v in resolved.items()},
        "package_versions": versions,
        "issues": issues,
    }
    
    logs_dir = resolved.get("logs_dir", repo_root / "logs")
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"setup_report_{ts}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
    #short report
    print ("BRANE sanity check ")
    print (f"Config: {config_path}")
    print (f"issues: {len(issues)}")
    for issue in issues:
            print (f" - {issue}")
            
    return 1 if (args.strict and issues) else 0

if __name__ == "__main__":
    raise SystemExit(main())    