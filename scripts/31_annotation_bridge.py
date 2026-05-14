#!/usr/bin/env python3
"""
Bootstrap script to generate inputs for Script 49b.

Generates:
    - results/20_node_annotation/01_input_data/annotation_bridge/node_table_base.csv
    from existing stage 10 NCBI node data + pair names
  
    - results/20_node_annotation/01_input_data/annotation_bridge/cancer_gene_flags.csv
    from existing stage 11 cancer gene annotation files

This script should be run ONCE before Script 49b.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_node_table_base() -> None:
    """
    Generate node_table_base.csv by collecting all per-pair node data.
    
    Structure:
      approved_symbol, entrez_id, ncbi_summary, pair_name
    """
    ncbi_dir = REPO_ROOT / "results/18_ncbi"
    
    if not ncbi_dir.exists():
        raise FileNotFoundError(f"NCBI gene fetch dir not found: {ncbi_dir}")
    
    results = []
    
    # Loop through all pair subdirectories
    for pair_dir in sorted(ncbi_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        
        pair_name = pair_dir.name
        node_file = pair_dir / f"{pair_name}_pair_nodes_ncbi_annotated.csv"
        
        if not node_file.exists():
            print(f"⚠️  SKIP: Node file not found: {node_file}")
            continue
        
        print(f"📖 Loading nodes from {pair_name}...")
        try:
            df = pd.read_csv(node_file)
            
            # Extract required columns and add pair_name
            if "approved_symbol" in df.columns and "entrez_id" in df.columns:
                subset = df[[
                    "approved_symbol",
                    "entrez_id",
                    "summary" if "summary" in df.columns else "ncbi_symbol"
                ]].copy()
                subset.columns = ["approved_symbol", "entrez_id", "ncbi_summary"]
                subset["pair_name"] = pair_name
                results.append(subset)
            else:
                print(f"⚠️  SKIP: Missing required columns in {pair_name}")
        except Exception as e:
            print(f"⚠️  ERROR loading {pair_name}: {e}")
    
    if not results:
        raise ValueError("No node data collected!")
    
    base_df = pd.concat(results, ignore_index=True)
    
    # Remove duplicates (keep first occurrence per gene-pair)
    base_df = base_df.drop_duplicates(subset=["approved_symbol", "pair_name"], keep="first")
    
    # Save
    output_path = REPO_ROOT / "results/20_node_annotation/01_input_data/annotation_bridge/node_table_base.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved node_table_base.csv: {len(base_df)} rows")
    print(f"  Path: {output_path}")


def bootstrap_cancer_gene_flags() -> None:
    """
    Generate cancer_gene_flags.csv by consolidating per-pair cancer gene annotations.
    
    Takes all unique genes and their cancer-gene metadata, deduplicating by approved_symbol.
    """
    cancer_annot_dir = REPO_ROOT / "results/19_cancer_gene_annotation"
    
    if not cancer_annot_dir.exists():
        raise FileNotFoundError(f"Cancer gene annotation dir not found: {cancer_annot_dir}")
    
    all_genes = {}  # approved_symbol → full row dict
    
    # Loop through all pair subdirectories
    for pair_dir in sorted(cancer_annot_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        
        pair_name = pair_dir.name
        cancer_file = pair_dir / f"{pair_name}_cancer_gene_master_annotation.csv"
        
        if not cancer_file.exists():
            print(f"⚠️  SKIP: Cancer annotation file not found: {cancer_file}")
            continue
        
        print(f"📖 Loading cancer genes from {pair_name}...")
        try:
            df = pd.read_csv(cancer_file)
            
            for _, row in df.iterrows():
                approved_symbol = row.get("approved_symbol")
                if pd.notna(approved_symbol) and approved_symbol not in all_genes:
                    # Store all columns for this gene on first encounter
                    all_genes[approved_symbol] = row.to_dict()
        except Exception as e:
            print(f"⚠️  ERROR loading {pair_name}: {e}")
    
    if not all_genes:
        raise ValueError("No cancer gene data collected!")
    
    # Convert to DataFrame
    cancer_df = pd.DataFrame.from_dict(all_genes, orient="index")
    cancer_df = cancer_df.reset_index(drop=True)
    
    # Save
    output_path = REPO_ROOT / "results/20_node_annotation/01_input_data/annotation_bridge/cancer_gene_flags.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cancer_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved cancer_gene_flags.csv: {len(cancer_df)} rows")
    print(f"  Path: {output_path}")
    print(f"  Columns: {list(cancer_df.columns)}")


def main():
    print("=" * 70)
    print("Bootstrap: Generate inputs for Script 49b")
    print("=" * 70)
    
    print("\n🔄 Step 1: Generate node_table_base.csv...")
    try:
        bootstrap_node_table_base()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise
    
    print("\n🔄 Step 2: Generate cancer_gene_flags.csv...")
    try:
        bootstrap_cancer_gene_flags()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise
    
    print("\n" + "=" * 70)
    print("✓ Bootstrap complete! Inputs ready for Script 49b")
    print("=" * 70)


if __name__ == "__main__":
    main()
