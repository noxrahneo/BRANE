#!/usr/bin/env python3
"""Consolidate HGNC/NCBI and cancer gene annotation tables into bridge inputs for 34_node_annotation_assembly.py."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_node_table_base() -> None:
    #collect per-pair ncbi-annotated node files and concatenate into node_table_base.csv
    ncbi_dir = REPO_ROOT / "results/18_ncbi"

    if not ncbi_dir.exists():
        raise FileNotFoundError(f"NCBI gene fetch dir not found: {ncbi_dir}")

    results = []

    for pair_dir in sorted(ncbi_dir.iterdir()):
        if not pair_dir.is_dir():
            continue

        pair_name = pair_dir.name
        node_file = pair_dir / f"{pair_name}_pair_nodes_ncbi_annotated.csv"

        if not node_file.exists():
            print(f"[SKIP] node file not found: {node_file}")
            continue

        print(f"loading nodes from {pair_name}...")
        try:
            df = pd.read_csv(node_file)
            #extract required columns and add pair_name
            if "approved_symbol" in df.columns and "entrez_id" in df.columns:
                summary_col = "summary" if "summary" in df.columns else "ncbi_symbol"
                subset = df[["approved_symbol", "entrez_id", summary_col]].copy()
                subset.columns = ["approved_symbol", "entrez_id", "ncbi_summary"]
                subset["pair_name"] = pair_name
                results.append(subset)
            else:
                print(f"[SKIP] missing required columns in {pair_name}")
        except Exception as e:
            print(f"[ERROR] loading {pair_name}: {e}")

    if not results:
        raise ValueError("No node data collected!")

    base_df = pd.concat(results, ignore_index=True)
    #deduplicate by gene+pair (keep first occurrence)
    base_df = base_df.drop_duplicates(subset=["approved_symbol", "pair_name"], keep="first")

    #save output
    output_path = REPO_ROOT / "results/20_node_annotation/01_input_data/annotation_bridge/node_table_base.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_path, index=False)

    print(f"saved node_table_base.csv: {len(base_df)} rows → {output_path}")


def bootstrap_cancer_gene_flags() -> None:
    #consolidate per-pair cancer gene annotations into cancer_gene_flags.csv; deduplicate by approved_symbol
    cancer_annot_dir = REPO_ROOT / "results/19_cancer_gene_annotation"

    if not cancer_annot_dir.exists():
        raise FileNotFoundError(f"Cancer gene annotation dir not found: {cancer_annot_dir}")

    all_genes: dict = {}  # approved_symbol → full row dict

    for pair_dir in sorted(cancer_annot_dir.iterdir()):
        if not pair_dir.is_dir():
            continue

        pair_name = pair_dir.name
        cancer_file = pair_dir / f"{pair_name}_cancer_gene_master_annotation.csv"

        if not cancer_file.exists():
            print(f"[SKIP] cancer annotation file not found: {cancer_file}")
            continue

        print(f"loading cancer genes from {pair_name}...")
        try:
            df = pd.read_csv(cancer_file)
            for _, row in df.iterrows():
                approved_symbol = row.get("approved_symbol")
                #store all columns for this gene on first encounter
                if pd.notna(approved_symbol) and approved_symbol not in all_genes:
                    all_genes[approved_symbol] = row.to_dict()
        except Exception as e:
            print(f"[ERROR] loading {pair_name}: {e}")

    if not all_genes:
        raise ValueError("No cancer gene data collected!")

    cancer_df = pd.DataFrame.from_dict(all_genes, orient="index").reset_index(drop=True)

    #save output
    output_path = REPO_ROOT / "results/20_node_annotation/01_input_data/annotation_bridge/cancer_gene_flags.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cancer_df.to_csv(output_path, index=False)

    print(f"saved cancer_gene_flags.csv: {len(cancer_df)} rows → {output_path}")


def main() -> int:
    #generate node_table_base.csv
    print("generating node_table_base.csv...")
    bootstrap_node_table_base()

    #generate cancer_gene_flags.csv
    print("generating cancer_gene_flags.csv...")
    bootstrap_cancer_gene_flags()

    print("Done. Bridge inputs ready for 34_node_annotation_assembly.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
