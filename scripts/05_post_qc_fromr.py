#!/usr/bin/env python3
"""Post-filter QC and pre/post comparison.

R_TRANSLATED: yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-filter QC")
    parser.add_argument("--cohort", default="results/bigboss_chopped.csv")
    parser.add_argument(
        "--filtered-dir",
        default="results/02_preprocess/01_filtered",
    )
    parser.add_argument(
        "--pre-summary",
        default="results/01_qc/qc_pre/pre_qc_summary.csv",
    )
    parser.add_argument(
        "--pre-per-cell",
        default="results/01_qc/qc_pre/pre_qc_per_cell.parquet",
    )
    parser.add_argument("--out-dir", default="results/01_qc/qc_post")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.qc_common import (  # pylint: disable=import-outside-toplevel
        load_per_cell_table,
        plot_pre_post_scatter,
        plot_sample_boxpanels,
        plot_sample_violinpanels,
        plot_pre_post_violinpanels,
        safe_path_component,
        save_per_cell_table,
        select_samples,
    )

    from utils.qc_functions import (  # pylint: disable=import-outside-toplevel
        compute_basic_qc,
        extract_per_cell_qc,
    )

    cohort_path = (repo_root / args.cohort).resolve()
    filtered_dir = (repo_root / args.filtered_dir).resolve()
    pre_summary_path = (repo_root / args.pre_summary).resolve()
    pre_per_cell_path = (repo_root / args.pre_per_cell).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    summary_rows: list[dict] = []
    per_cell_parts: list[pd.DataFrame] = []

    total = len(cohort)
    for pos, row in enumerate(cohort.itertuples(index=False), start=1):
        sample_name = str(row.SampleName)
        condition = str(row.Condition)
        h5ad_file = (
            filtered_dir
            / safe_path_component(condition)
            / f"{sample_name}_filtered.h5ad"
        )
        if not h5ad_file.exists():
            print(f"[{pos}/{total}] {sample_name}: filtered file missing")
            continue

        adata = sc.read_h5ad(h5ad_file)
        gene_names = pd.Series(adata.var_names.astype(str))
        summary = compute_basic_qc(adata.X, gene_names)
        summary.update(
            {
                "SampleName": sample_name,
                "Condition": condition,
                "stage": "post",
                "filtered_file": str(h5ad_file),
            }
        )
        summary_rows.append(summary)

        per_cell = extract_per_cell_qc(adata.X, gene_names)
        per_cell["SampleName"] = sample_name
        per_cell["Condition"] = condition
        per_cell["stage"] = "post"
        per_cell_parts.append(per_cell)
        print(f"[{pos}/{total}] {sample_name}: loaded {adata.n_obs} cells")

    if not summary_rows:
        print("ERROR: no filtered samples were processed")
        return 1

    post_summary = pd.DataFrame(summary_rows)
    post_per_cell = pd.concat(per_cell_parts, ignore_index=True)

    post_summary_file = out_dir / "post_qc_summary.csv"
    post_summary.to_csv(post_summary_file, index=False)
    per_cell_file = save_per_cell_table(post_per_cell, out_dir, "post_qc")
    post_boxplot_file = out_dir / "post_qc_boxplots.png"
    post_violin_file = out_dir / "post_qc_violins.png"
    plot_sample_boxpanels(
        post_per_cell,
        post_boxplot_file,
        title_suffix="post",
    )
    plot_sample_violinpanels(
        post_per_cell,
        post_violin_file,
        title_suffix="post",
    )

    compare_file = None
    compare_plot = None
    compare_violin = None
    if pre_summary_path.exists():
        pre_summary = pd.read_csv(pre_summary_path)
        compare_df = pre_summary.merge(
            post_summary,
            on=["SampleName", "Condition"],
            how="inner",
            suffixes=("_pre", "_post"),
        )
        if not compare_df.empty:
            compare_df["cell_retention_pct"] = (
                100.0 * compare_df["n_cells_post"] / compare_df["n_cells_pre"]
            )
            compare_df["median_genes_delta"] = (
                compare_df["median_genes_per_cell_post"]
                - compare_df["median_genes_per_cell_pre"]
            )
            compare_df["median_mito_delta"] = (
                compare_df["pct_mito_median_post"]
                - compare_df["pct_mito_median_pre"]
            )

            compare_file = out_dir / "pre_post_qc_comparison.csv"
            compare_df.to_csv(compare_file, index=False)
            compare_plot = out_dir / "pre_post_qc_scatter.png"
            plot_pre_post_scatter(compare_df, compare_plot)

    pre_per_cell = load_per_cell_table(pre_per_cell_path)
    if pre_per_cell is not None and not pre_per_cell.empty:
        pre_cols = {
            "SampleName",
            "Condition",
            "stage",
            "n_counts",
            "n_genes",
            "pct_mito",
        }
        if pre_cols.issubset(set(pre_per_cell.columns)):
            merged_per_cell = pd.concat(
                [pre_per_cell, post_per_cell],
                ignore_index=True,
            )
            compare_violin = out_dir / "pre_post_qc_violins.png"
            plot_pre_post_violinpanels(merged_per_cell, compare_violin)

    print("\nPost-QC complete")
    print(f"Samples processed: {len(post_summary)}")
    print(f"Saved: {post_summary_file}")
    print(f"Saved: {per_cell_file}")
    print(f"Saved: {post_boxplot_file}")
    print(f"Saved: {post_violin_file}")
    if compare_file is not None:
        print(f"Saved: {compare_file}")
    if compare_plot is not None:
        print(f"Saved: {compare_plot}")
    if compare_violin is not None:
        print(f"Saved: {compare_violin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
