#!/usr/bin/env python3
"""02_chopper.py: filtering bogboss.csv to the core cohort for downstream analysis"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter bigboss.csv to the core cohort for downstream analysis.")
    parser.add_argument("--input", default="results/00_data_ingestion/bigboss.csv", help="Path to the input bigboss.csv")
    parser.add_argument("--output", default="results/00_data_ingestion/bigboss_chopped.csv", help="Path to the output bigboss_chopped.csv")
    parser.add_argument("--keep-source", default="Total", help="Keep only this Source value (default: 'Total')")
    parser.add_argument("--keep-gender", default="Female", help="Keep only this Gender value (default: 'Female')")
    return parser.parse_args()

def apply_filters(df: pd.DataFrame, keep_source: str, keep_gender: str) -> pd.DataFrame:
    #keep total-cell smaples only
    out = df[df["Source"] == keep_source].copy()
    
    #keep female samples only
    out = out[out["Gender"] == keep_gender].copy()
    
    #remove lymph node condition
    out = out[out["Condition"] != "Involved LN"].copy()
    
    #keep only study conditions of interest 
    keep_conditions = {
        "Normal",
        "Normal BRCA1+/- pre-neoplastic",
        "Triple negative tumor",
        "Triple negative (BRCA1) tumor",
        "HER2+ tumor",
        "ER+ tumor",
    }
    out = out[out["Condition"].isin(keep_conditions)].copy()

    return out

def print_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print("\n=== Chopper summary ===")
    print(f"Rows before: {len(before)}")
    print(f"Rows after:  {len(after)}")
    print("\nBy condition:")
    print(after["Condition"].value_counts(dropna=False).to_string())
    print("\nBy source:")
    print(after["Source"].value_counts(dropna=False).to_string())
    print("\nBy gender:")
    print(after["Gender"].value_counts(dropna=False).to_string())


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    in_path = (repo_root / args.input).resolve()
    out_path = (repo_root / args.output).resolve()

    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}")
        return 1

    df = pd.read_csv(in_path)
    chopped = apply_filters(df, keep_source=args.keep_source, keep_gender=args.keep_gender)

    if chopped.empty:
        print("ERROR: filtering produced 0 rows. Check filter values.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chopped.to_csv(out_path, index=False)

    print_summary(df, chopped)  
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())