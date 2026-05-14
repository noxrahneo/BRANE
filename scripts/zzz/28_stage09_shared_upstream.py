#!/usr/bin/env python3
# flake8: noqa: E501
"""Stage-09 shared upstream builder.

Per condition-pair, this script creates the shared artifacts used by both
branch A (scale-free) and branch B (permutation thresholding):
- DEG lists (up/down)
- per-gene DEG annotation table
- raw C/S/D table with denominator fixed to one
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.network_utils import (
    align_corrs,
    build_gene_change_table,
    compute_allvalues,
    list_conditions,
    load_corr_payload,
    load_deg_stats,
    load_n_profiles,
    parse_pairs,
    pick_degs,
    resolve_base,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-09 shared upstream artifacts")
    parser.add_argument(
        "--corr-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition *_pearson_corr.npz",
    )
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Root with <condition>_pseudobulk_logcpm.h5ad",
    )
    parser.add_argument(
        "--deg-dir",
        default="results/10_deg_ttest",
        help="Root with contrast DEG tables",
    )
    parser.add_argument(
        "--output-dir",
        default="results/14_differential_prep",
        help="Output directory for shared stage-09 artifacts",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional pair case:control. Repeat for multiple",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="DEG threshold on FDR",
    )
    parser.add_argument(
        "--min-abs-log2fc",
        type=float,
        default=0.2,
        help="DEG threshold on |log2FC|",
    )
    parser.add_argument(
        "--use-degs",
        action="store_true",
        help="Keep only DEGs for downstream network branch inputs",
    )
    return parser.parse_args()


def run_pair(
    case: str,
    ctrl: str,
    corr_root: Path,
    expr_root: Path,
    deg_root: Path,
    out_root: Path,
    fdr_threshold: float,
    min_abs_log2fc: float,
    use_degs: bool,
) -> dict[str, object]:
    pair_name = f"{case}__vs__{ctrl}"
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    corr_case, genes_case, case_corr_file = load_corr_payload(corr_root, case)
    corr_ctrl, genes_ctrl, ctrl_corr_file = load_corr_payload(corr_root, ctrl)
    corr_case, corr_ctrl, genes_shared = align_corrs(corr_case, genes_case, corr_ctrl, genes_ctrl)

    n_case_profiles = load_n_profiles(expr_root, case)
    n_ctrl_profiles = load_n_profiles(expr_root, ctrl)

    deg_df = load_deg_stats(deg_root, case, ctrl)
    deg_all, up_degs, down_degs = pick_degs(deg_df, fdr_threshold=fdr_threshold, min_abs_log2fc=min_abs_log2fc)

    (pair_out / "up_degs.txt").write_text("\n".join(up_degs) + ("\n" if up_degs else ""), encoding="utf-8")
    (pair_out / "down_degs.txt").write_text("\n".join(down_degs) + ("\n" if down_degs else ""), encoding="utf-8")

    if use_degs:
        genes_keep = np.array([g for g in genes_shared if g in deg_all], dtype=str)
        if genes_keep.size < 3:
            print(f"[warn] {pair_name}: too few DEGs ({genes_keep.size}), fallback to shared genes")
            genes_keep = genes_shared.copy()
    else:
        genes_keep = genes_shared.copy()

    gene_change_df = build_gene_change_table(
        deg_df=deg_df,
        genes_keep=genes_keep,
        fdr_threshold=fdr_threshold,
        min_abs_log2fc=min_abs_log2fc,
    )
    gene_change_file = pair_out / f"{pair_name}_gene_expression_change.csv"
    gene_change_df.to_csv(gene_change_file, index=False)

    (pair_out / f"{pair_name}_genes_keep.txt").write_text(
        "\n".join(genes_keep.tolist()) + ("\n" if genes_keep.size > 0 else ""),
        encoding="utf-8",
    )

    idx_map = {g: i for i, g in enumerate(genes_shared)}
    ig = np.array([idx_map[g] for g in genes_keep], dtype=int)
    r_case = corr_case[np.ix_(ig, ig)]
    r_ctrl = corr_ctrl[np.ix_(ig, ig)]

    all_values = compute_allvalues(r_case, r_ctrl, genes_keep)
    all_values_file = pair_out / f"{pair_name}_AllValues.tsv"
    all_values.to_csv(all_values_file, sep="\t", index=False)

    summary = {
        "pair": pair_name,
        "case": case,
        "control": ctrl,
        "use_degs": bool(use_degs),
        "fdr_threshold": float(fdr_threshold),
        "min_abs_log2fc": float(min_abs_log2fc),
        "n_case_profiles": int(n_case_profiles),
        "n_control_profiles": int(n_ctrl_profiles),
        "n_genes_shared": int(genes_shared.size),
        "n_genes_used": int(genes_keep.size),
        "n_edges_allvalues": int(all_values.shape[0]),
        "n_degs_total": int(len(deg_all)),
        "n_up_degs": int(len(up_degs)),
        "n_down_degs": int(len(down_degs)),
        "source_case_corr": str(case_corr_file),
        "source_control_corr": str(ctrl_corr_file),
        "all_values_file": str(all_values_file),
        "gene_change_file": str(gene_change_file),
    }
    save_json(pair_out / f"{pair_name}_shared_summary.json", summary)

    print(f"[{pair_name}] genes={summary['n_genes_used']} edges={summary['n_edges_allvalues']}")
    return summary


def main() -> None:
    args = parse_args()
    corr_root = resolve_base(args.corr_dir)
    expr_root = resolve_base(args.expr_dir)
    deg_root = resolve_base(args.deg_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = list_conditions(corr_root)
    pairs = parse_pairs(args.pair, conditions)

    summaries: list[dict[str, object]] = []
    for case, ctrl in pairs:
        summaries.append(
            run_pair(
                case=case,
                ctrl=ctrl,
                corr_root=corr_root,
                expr_root=expr_root,
                deg_root=deg_root,
                out_root=out_root,
                fdr_threshold=float(args.fdr_threshold),
                min_abs_log2fc=float(args.min_abs_log2fc),
                use_degs=bool(args.use_degs),
            )
        )

    pd.DataFrame(summaries).sort_values("pair").to_csv(out_root / "shared_upstream_summary.csv", index=False)
    print(f"Done. Stage-09 shared upstream outputs: {out_root}")


if __name__ == "__main__":
    main()
