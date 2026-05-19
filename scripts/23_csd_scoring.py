#!/usr/bin/env python3
# flake8: noqa: E501
"""Build per-condition-pair CSD scoring artifacts for network construction.

Outputs per pair:
- DEG lists (up/down)
- per-gene DEG annotation table
- raw C/S/D AllValues table
Supports shared-gene or union-gene universes.
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


def align_corrs_union(
    corr_case: np.ndarray,
    genes_case: np.ndarray,
    corr_ctrl: np.ndarray,
    genes_ctrl: np.ndarray,
    fill_missing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #align case/ctrl matrices on gene union; missing correlations filled with fill_missing
    case_list = [str(g) for g in genes_case]
    union_list = case_list + [str(g) for g in genes_ctrl if str(g) not in set(case_list)]
    genes_union = np.asarray(union_list, dtype=str)

    idx_case = {str(g): i for i, g in enumerate(genes_case)}
    idx_ctrl = {str(g): i for i, g in enumerate(genes_ctrl)}

    n = genes_union.size
    case_u = np.full((n, n), float(fill_missing), dtype=np.float64)
    ctrl_u = np.full((n, n), float(fill_missing), dtype=np.float64)

    case_present = [i for i, g in enumerate(genes_union) if g in idx_case]
    ctrl_present = [i for i, g in enumerate(genes_union) if g in idx_ctrl]

    if case_present:
        orig_case = np.array([idx_case[genes_union[i]] for i in case_present], dtype=int)
        case_u[np.ix_(case_present, case_present)] = corr_case[np.ix_(orig_case, orig_case)]
    if ctrl_present:
        orig_ctrl = np.array([idx_ctrl[genes_union[i]] for i in ctrl_present], dtype=int)
        ctrl_u[np.ix_(ctrl_present, ctrl_present)] = corr_ctrl[np.ix_(orig_ctrl, orig_ctrl)]

    np.fill_diagonal(case_u, 1.0)
    np.fill_diagonal(ctrl_u, 1.0)
    return case_u, ctrl_u, genes_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-pair CSD scoring artifacts")
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
        default="results/12_csd_scoring",
        help="Output directory for union-capable stage-09 artifacts",
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
    parser.add_argument(
        "--gene-universe",
        choices=["union", "shared"],
        default="union",
        help="Gene universe to use when building pairwise C/S/D values",
    )
    parser.add_argument(
        "--fill-missing-corr",
        type=float,
        default=0.0,
        help="Correlation fill value for missing genes in union mode",
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
    gene_universe: str,
    fill_missing_corr: float,
) -> dict[str, object]:
    pair_name = f"{case}__vs__{ctrl}"
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    #load correlation matrices for case and control
    corr_case_raw, genes_case, case_corr_file = load_corr_payload(corr_root, case)
    corr_ctrl_raw, genes_ctrl, ctrl_corr_file = load_corr_payload(corr_root, ctrl)

    shared_set = set(str(g) for g in genes_case).intersection(set(str(g) for g in genes_ctrl))
    n_genes_shared = len(shared_set)

    #align matrices on shared or union gene universe
    if gene_universe == "shared":
        corr_case, corr_ctrl, genes_universe = align_corrs(corr_case_raw, genes_case, corr_ctrl_raw, genes_ctrl)
    else:
        corr_case, corr_ctrl, genes_universe = align_corrs_union(
            corr_case=corr_case_raw, genes_case=genes_case,
            corr_ctrl=corr_ctrl_raw, genes_ctrl=genes_ctrl,
            fill_missing=float(fill_missing_corr),
        )

    #tag each gene as shared, case_only, or ctrl_only
    case_set = {str(g) for g in genes_case}
    ctrl_set_local = {str(g) for g in genes_ctrl}
    presence_rows = []
    for g in genes_universe:
        in_case = g in case_set
        in_ctrl = g in ctrl_set_local
        presence = "shared" if (in_case and in_ctrl) else ("case_only" if in_case else "ctrl_only")
        presence_rows.append({"gene": g, "presence": presence})
    gene_presence_df = pd.DataFrame(presence_rows)
    gene_presence_df.to_csv(pair_out / f"{pair_name}_gene_presence.csv", index=False)

    n_case_profiles = load_n_profiles(expr_root, case)
    n_ctrl_profiles = load_n_profiles(expr_root, ctrl)

    #load deg stats and write up/down gene lists
    deg_df = load_deg_stats(deg_root, case, ctrl)
    deg_all, up_degs, down_degs = pick_degs(deg_df, fdr_threshold=fdr_threshold, min_abs_log2fc=min_abs_log2fc)
    (pair_out / "up_degs.txt").write_text("\n".join(up_degs) + ("\n" if up_degs else ""), encoding="utf-8")
    (pair_out / "down_degs.txt").write_text("\n".join(down_degs) + ("\n" if down_degs else ""), encoding="utf-8")

    #select genes to use: degs only or full universe
    if use_degs:
        genes_keep = np.array([g for g in genes_universe if g in deg_all], dtype=str)
        if genes_keep.size < 3:
            print(f"[warn] {pair_name}: too few DEGs ({genes_keep.size}), fallback to shared genes")
            genes_keep = genes_universe.copy()
    else:
        genes_keep = genes_universe.copy()

    #build per-gene expression change table and subset correlation matrices
    gene_change_df = build_gene_change_table(deg_df=deg_df, genes_keep=genes_keep,
                                              fdr_threshold=fdr_threshold, min_abs_log2fc=min_abs_log2fc)
    gene_change_file = pair_out / f"{pair_name}_gene_expression_change.csv"
    gene_change_df.to_csv(gene_change_file, index=False)
    (pair_out / f"{pair_name}_genes_keep.txt").write_text(
        "\n".join(genes_keep.tolist()) + ("\n" if genes_keep.size > 0 else ""), encoding="utf-8")

    idx_map = {g: i for i, g in enumerate(genes_universe)}
    ig = np.array([idx_map[g] for g in genes_keep], dtype=int)
    r_case = corr_case[np.ix_(ig, ig)]
    r_ctrl = corr_ctrl[np.ix_(ig, ig)]

    #compute raw csd allvalues table
    all_values = compute_allvalues(r_case, r_ctrl, genes_keep)
    all_values_file = pair_out / f"{pair_name}_AllValues.tsv"
    all_values.to_csv(all_values_file, sep="\t", index=False)

    summary = {
        "pair": pair_name,
        "case": case,
        "control": ctrl,
        "gene_universe": str(gene_universe),
        "fill_missing_corr": float(fill_missing_corr),
        "use_degs": bool(use_degs),
        "fdr_threshold": float(fdr_threshold),
        "min_abs_log2fc": float(min_abs_log2fc),
        "n_case_profiles": int(n_case_profiles),
        "n_control_profiles": int(n_ctrl_profiles),
        "n_genes_case": int(len(genes_case)),
        "n_genes_control": int(len(genes_ctrl)),
        "n_genes_shared": int(n_genes_shared),
        "n_genes_case_only": int((gene_presence_df["presence"] == "case_only").sum()),
        "n_genes_ctrl_only": int((gene_presence_df["presence"] == "ctrl_only").sum()),
        "n_genes_universe": int(genes_universe.size),
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


def main() -> int:
    args = parse_args()

    #resolve paths and list available conditions
    corr_root = resolve_base(args.corr_dir)
    expr_root = resolve_base(args.expr_dir)
    deg_root = resolve_base(args.deg_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    #parse condition pairs to process
    conditions = list_conditions(corr_root)
    pairs = parse_pairs(args.pair, conditions)

    #run csd scoring for each pair
    summaries: list[dict[str, object]] = []
    for case, ctrl in pairs:
        summaries.append(run_pair(
            case=case, ctrl=ctrl, corr_root=corr_root, expr_root=expr_root,
            deg_root=deg_root, out_root=out_root,
            fdr_threshold=float(args.fdr_threshold), min_abs_log2fc=float(args.min_abs_log2fc),
            use_degs=bool(args.use_degs), gene_universe=str(args.gene_universe),
            fill_missing_corr=float(args.fill_missing_corr),
        ))

    #write combined summary csv
    pd.DataFrame(summaries).sort_values("pair").to_csv(out_root / "shared_upstream_summary.csv", index=False)
    print(f"Done. CSD scoring outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
