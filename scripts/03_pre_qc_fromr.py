#!/usr/bin/env python3
"""Pre-filter QC on raw matrices.

R_TRANSLATED: yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-filter QC")
    parser.add_argument("--cohort", default="results/00_data_ingestion/bigboss_chopped.csv")
    parser.add_argument("--data-dir", default="devil_fruit/GSE161529_RAW")
    parser.add_argument("--features", default="devil_fruit/GSE161529_features.tsv")
    parser.add_argument("--out-dir", default="results/01_qc/qc_pre")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    #add repo root to sys.path so utils packages resolve correctly
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.qc_common import (  # pylint: disable=import-outside-toplevel
        plot_sample_boxpanels,
        plot_sample_violinpanels,
        save_per_cell_table,
        select_samples,
    )

    from utils.qc_functions import (  # pylint: disable=import-outside-toplevel
        compute_basic_qc,
        extract_per_cell_qc,
        load_features,
        load_matrix_and_barcodes,
    )

    #resolve paths and create output dir
    cohort_path = (repo_root / args.cohort).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    features_path = (repo_root / args.features).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    #load and subset cohort
    if not cohort_path.exists():
        print(f"ERROR: cohort file not found: {cohort_path}")
        return 1

    cohort = pd.read_csv(cohort_path)
    cohort = select_samples(cohort, args.condition)
    if args.max_samples > 0:
        cohort = cohort.head(args.max_samples).copy()
    if cohort.empty:
        print(f"ERROR: no samples for condition='{args.condition}'")
        return 1

    #load shared gene name list from features file
    gene_names = load_features(features_path)["gene_name"]
    summary_rows: list[dict] = []
    per_cell_parts: list[pd.DataFrame] = []

    #compute pre-filter qc metrics for each sample
    total = len(cohort)
    for pos, row in enumerate(cohort.itertuples(index=False), start=1):
        sample_name = str(row.SampleName)
        matrix_file = data_dir / str(row.MatrixFile)
        barcodes_file = data_dir / str(row.BarcodesFile)
        if not matrix_file.exists() or not barcodes_file.exists():
            print(f"[{pos}/{total}] {sample_name}: missing files, skipping")
            continue

        matrix, _ = load_matrix_and_barcodes(matrix_file, barcodes_file)
        summary = compute_basic_qc(matrix, gene_names)
        summary.update({
            "SampleName": sample_name,
            "Condition": str(row.Condition),
            "stage": "pre",
        })
        summary_rows.append(summary)

        per_cell = extract_per_cell_qc(matrix, gene_names)
        per_cell["SampleName"] = sample_name
        per_cell["Condition"] = str(row.Condition)
        per_cell["stage"] = "pre"
        per_cell_parts.append(per_cell)
        print(f"[{pos}/{total}] {sample_name}: loaded {matrix.shape[0]} cells")

    if not summary_rows:
        print("ERROR: no samples were processed")
        return 1

    #concatenate results across samples
    summary_df = pd.DataFrame(summary_rows)
    per_cell_df = pd.concat(per_cell_parts, ignore_index=True)

    #save summary csv, per-cell table, and qc plots
    summary_file = out_dir / "pre_qc_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    per_cell_file = save_per_cell_table(per_cell_df, out_dir, "pre_qc")
    boxplot_file = out_dir / "pre_qc_boxplots.png"
    violin_file = out_dir / "pre_qc_violins.png"
    plot_sample_boxpanels(per_cell_df, boxplot_file, title_suffix="pre")
    plot_sample_violinpanels(per_cell_df, violin_file, title_suffix="pre")

    print("\nPre-QC complete")
    print(f"Samples processed: {len(summary_df)}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {per_cell_file}")
    print(f"Saved: {boxplot_file}")
    print(f"Saved: {violin_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
