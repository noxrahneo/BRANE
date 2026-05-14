#!/usr/bin/env python3
"""Filter cells for one or all cohort conditions.

R_TRANSLATED: yes

Based on filtering logic from `NormTotal.R`:
- Cell filtering by per-sample thresholds (Mito, GeneLower, GeneUpper, LibSize)
- Basic gene filtering (>=1% cells, valid symbol, unique symbol)
- Save per-sample filtered `.h5ad` files + a run summary CSV
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter cohort samples")
    parser.add_argument(
        "--cohort",
        default="results/00_data_ingestion/bigboss_chopped.csv",
        help="Input cohort CSV",
    )
    parser.add_argument(
        "--data-dir",
        default="devil_fruit/GSE161529_RAW",
        help="Directory containing matrix/barcode files",
    )
    parser.add_argument(
        "--features",
        default="devil_fruit/GSE161529_features.tsv",
        help="Global features TSV",
    )
    parser.add_argument(
        "--out-dir",
        default="results/02_preprocess/01_filtered",
        help="Output directory",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print all available conditions from cohort and exit",
    )
    return parser.parse_args()


def safe_path_component(value: str) -> str:
    """Convert arbitrary label into filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def get_available_conditions(cohort: pd.DataFrame) -> list[str]:
    return sorted(set(cohort["Condition"].astype(str).str.strip()))


def suggest_conditions(requested: str, options: list[str]) -> list[str]:
    req = requested.strip()
    close = difflib.get_close_matches(req, options, n=5, cutoff=0.45)
    contains = [opt for opt in options if req.lower() in opt.lower()]
    suggestions: list[str] = []
    for item in close + contains:
        if item not in suggestions:
            suggestions.append(item)
    return suggestions[:5]


def select_samples(cohort: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition.lower() == "all":
        return cohort.copy()
    return cohort[cohort["Condition"].astype(str) == condition].copy()


def load_one_sample(
    matrix_file: Path,
    barcodes_file: Path,
    features_file: Path,
) -> sc.AnnData:
    """Load one sample matrix as AnnData (cells x genes)."""
    with gzip.open(matrix_file, "rb") as handle:
        matrix = mmread(handle).T.tocsr()

    barcodes = pd.read_csv(
        barcodes_file,
        header=None,
        names=["barcode"],
        compression="gzip",
    )

    features = pd.read_csv(features_file, sep="\t", header=None)
    features = features.iloc[:, :2].copy()
    features.columns = ["gene_id", "gene_name"]

    if matrix.shape[0] != len(barcodes):
        raise ValueError("Matrix/barcode mismatch")
    if matrix.shape[1] != len(features):
        raise ValueError("Matrix/features mismatch")

    adata = sc.AnnData(X=matrix)
    adata.obs_names = barcodes["barcode"].astype(str).values
    adata.var_names = features["gene_id"].astype(str).values
    adata.var["gene_id"] = features["gene_id"].astype(str).values
    adata.var["gene_name"] = features["gene_name"].astype(str).values
    return adata


def add_cell_qc_metrics(adata: sc.AnnData) -> None:
    """Compute per-cell metrics used by threshold filtering."""
    adata.obs["lib_size"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    mito_mask = adata.var["gene_name"].str.upper().str.startswith("MT-").values
    if mito_mask.any():
        mito_counts = np.asarray(adata[:, mito_mask].X.sum(axis=1)).ravel()
        adata.obs["percent_mito"] = np.divide(
            mito_counts,
            adata.obs["lib_size"].values,
            out=np.zeros_like(mito_counts, dtype=float),
            where=adata.obs["lib_size"].values > 0,
        )
    else:
        adata.obs["percent_mito"] = 0.0


def filter_cells(
    adata: sc.AnnData,
    mito_upper: float,
    n_genes_lower: float,
    n_genes_upper: float,
    lib_upper: float,
) -> sc.AnnData:
    keep = (
        (adata.obs["percent_mito"] < float(mito_upper))
        & (adata.obs["n_genes"] > float(n_genes_lower))
        & (adata.obs["n_genes"] < float(n_genes_upper))
        & (adata.obs["lib_size"] < float(lib_upper))
    )
    return adata[keep.values].copy()


def filter_genes(adata: sc.AnnData) -> sc.AnnData:
    n_cells = adata.n_obs
    min_cells = max(1, int(np.ceil(n_cells * 0.01)))

    detected_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep_expr = detected_cells >= min_cells

    gene_name = adata.var["gene_name"].astype(str)
    keep_valid = gene_name.notna() & (gene_name != "") & (gene_name != "nan")
    keep_unique = ~gene_name.duplicated()

    keep = keep_expr & keep_valid.values & keep_unique.values
    out = adata[:, keep].copy()
    out.var_names = out.var["gene_name"].astype(str).values
    return out


def save_filtered_sample(
    adata: sc.AnnData,
    sample_name: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sample_name}_filtered.h5ad"
    adata.write_h5ad(out_file, compression="gzip")
    return out_file


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cohort_path = (repo_root / args.cohort).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    features_path = (repo_root / args.features).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cohort_path.exists():
        print(f"ERROR: Cohort file not found: {cohort_path}")
        return 1

    cohort = pd.read_csv(cohort_path)
    required_cols = [
        "SampleName",
        "Condition",
        "MatrixFile",
        "BarcodesFile",
        "Mito",
        "GeneLower",
        "GeneUpper",
        "LibSize",
    ]
    missing = [col for col in required_cols if col not in cohort.columns]
    if missing:
        print(f"ERROR: Missing required cohort columns: {missing}")
        return 1

    options = get_available_conditions(cohort)
    if args.list_conditions:
        print("Available conditions:")
        for name in options:
            print(f"- {name}")
        return 0

    selected = select_samples(cohort, args.condition)
    if selected.empty:
        print(f"ERROR: No samples found for condition='{args.condition}'.")
        suggestions = suggest_conditions(args.condition, options)
        if suggestions:
            print("Did you mean:")
            for name in suggestions:
                print(f"- {name}")
        print("Tip: use --list-conditions to view exact labels.")
        return 1

    summary_rows = []
    seen_condition_map: set[tuple[str, str]] = set()

    for idx, row in selected.iterrows():
        sample_name = str(row["SampleName"])
        condition = str(row["Condition"])
        matrix_file = data_dir / str(row["MatrixFile"])
        barcodes_file = data_dir / str(row["BarcodesFile"])

        if not matrix_file.exists() or not barcodes_file.exists():
            print(f"[{idx}] {sample_name}: missing matrix/barcodes, skipping")
            continue

        print(f"Processing {sample_name} ({condition})...")

        try:
            adata = load_one_sample(matrix_file, barcodes_file, features_path)
            n_cells_before, n_genes_before = adata.n_obs, adata.n_vars

            add_cell_qc_metrics(adata)
            adata = filter_cells(
                adata=adata,
                mito_upper=row["Mito"],
                n_genes_lower=row["GeneLower"],
                n_genes_upper=row["GeneUpper"],
                lib_upper=row["LibSize"],
            )
            adata = filter_genes(adata)
            n_cells_after, n_genes_after = adata.n_obs, adata.n_vars

            safe_condition = safe_path_component(condition)
            map_key = (condition, safe_condition)
            if map_key not in seen_condition_map:
                print(f"Output mapping: '{condition}' -> '{safe_condition}'")
                seen_condition_map.add(map_key)

            condition_dir = out_dir / safe_condition
            out_file = save_filtered_sample(adata, sample_name, condition_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[{idx}] {sample_name}: failed ({exc})")
            continue

        summary_rows.append(
            {
                "SampleName": sample_name,
                "Condition": condition,
                "cells_before": n_cells_before,
                "cells_after": n_cells_after,
                "genes_before": n_genes_before,
                "genes_after": n_genes_after,
                "retained_pct": (
                    100.0 * n_cells_after / n_cells_before
                    if n_cells_before > 0
                    else np.nan
                ),
                "output_file": str(out_file),
            }
        )

    if not summary_rows:
        print("ERROR: No samples were processed.")
        return 1

    summary = pd.DataFrame(summary_rows)
    summary_name = "filtering_summary.csv"
    if args.condition.lower() != "all":
        safe_condition = safe_path_component(args.condition)
        summary_name = f"filtering_summary_{safe_condition}.csv"
    summary_file = out_dir / summary_name
    summary.to_csv(summary_file, index=False)

    print("\nFiltering complete.")
    print(f"Samples processed: {len(summary_rows)}")
    print(f"Summary: {summary_file}")
    print(f"Filtered samples directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
