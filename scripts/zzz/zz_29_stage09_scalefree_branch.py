#!/usr/bin/env python3
# flake8: noqa: E501
"""Stage-09 branch A: scale-free thresholding on shared upstream artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from utils.network_utils import (
    adjacency_from_delta,
    align_corrs,
    augment_edges_with_wto,
    choose_power,
    classify_edges_no_threshold,
    compute_node_and_network_metrics,
    export_csd_files,
    list_conditions,
    load_corr_payload,
    parse_pairs,
    parse_powers,
    resolve_base,
    save_json,
    scale_free_fit_from_connectivity,
    top_edges_from_adjacency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-09 scale-free branch")
    parser.add_argument(
        "--corr-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition *_pearson_corr.npz",
    )
    parser.add_argument(
        "--shared-dir",
        default="results/14_differential_prep",
        help="Shared upstream output directory from script 40",
    )
    parser.add_argument(
        "--output-dir",
        default="results/15_scalefree_networks",
        help="Output directory for stage-09 branch A",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional pair case:control. Repeat for multiple",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned"],
        default="signed",
        help="Adjacency transform type for differential matrix",
    )
    parser.add_argument(
        "--delta-scale",
        choices=["half", "none"],
        default="half",
        help="Scale delta by 1/2 before adjacency",
    )
    parser.add_argument(
        "--powers",
        default="1,2,3,4,5,6,7,8,9,10,12,14,16,18,20",
        help="Comma-separated soft-threshold powers",
    )
    parser.add_argument(
        "--target-signed-r2",
        type=float,
        default=0.70,
        help="Target signed R^2 for power selection",
    )
    parser.add_argument(
        "--min-mean-connectivity",
        type=float,
        default=5.0,
        help="Minimum mean connectivity",
    )
    parser.add_argument(
        "--r2-plateau-delta",
        type=float,
        default=0.03,
        help="Plateau tolerance for beta selection",
    )
    parser.add_argument(
        "--r2-near-best-delta",
        type=float,
        default=0.02,
        help="Fallback tolerance near best R^2",
    )
    parser.add_argument(
        "--degree-bins",
        type=int,
        default=20,
        help="Histogram bins for scale-free fit",
    )
    parser.add_argument(
        "--force-power",
        type=int,
        default=None,
        help="Force beta power",
    )
    parser.add_argument(
        "--candidate-min-weight",
        type=float,
        default=0.01,
        help="Adjacency pre-filter before edge export",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=50,
        help="Top hubs to export",
    )
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=0,
        help="Optional analysis cap: keep top-K edges by selected_value (0 keeps all)",
    )
    return parser.parse_args()


def plot_soft_threshold(scan_df: pd.DataFrame, target_r2: float, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    x = scan_df["power"].to_numpy()
    y1 = scan_df["signed_r2"].to_numpy()
    y2 = scan_df["mean_connectivity"].to_numpy()

    axes[0].plot(x, y1, marker="o")
    axes[0].axhline(float(target_r2), color="red", linestyle="--", linewidth=1.3)
    axes[0].set_xlabel("Power")
    axes[0].set_ylabel("Scale-free fit (signed R^2)")
    axes[0].set_title(f"{title}: scale-free fit vs power")

    axes[1].plot(x, y2, marker="o")
    axes[1].set_xlabel("Power")
    axes[1].set_ylabel("Mean connectivity")
    axes[1].set_title(f"{title}: mean connectivity vs power")

    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def run_pair(
    case: str,
    ctrl: str,
    corr_root: Path,
    shared_root: Path,
    out_root: Path,
    powers: list[int],
    args: argparse.Namespace,
) -> dict[str, object]:
    pair_name = f"{case}__vs__{ctrl}"
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    shared_pair_dir = shared_root / pair_name
    all_values_file = shared_pair_dir / f"{pair_name}_AllValues.tsv"
    genes_keep_file = shared_pair_dir / f"{pair_name}_genes_keep.txt"
    gene_change_file = shared_pair_dir / f"{pair_name}_gene_expression_change.csv"

    if not all_values_file.exists() or not genes_keep_file.exists() or not gene_change_file.exists():
        raise FileNotFoundError(f"Missing shared artifacts for pair {pair_name} in {shared_pair_dir}")

    genes_keep = np.array(
        [line.strip() for line in genes_keep_file.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=str,
    )

    corr_case, genes_case, case_corr_file = load_corr_payload(corr_root, case)
    corr_ctrl, genes_ctrl, ctrl_corr_file = load_corr_payload(corr_root, ctrl)
    corr_case, corr_ctrl, genes_shared = align_corrs(corr_case, genes_case, corr_ctrl, genes_ctrl)

    idx_map = {g: i for i, g in enumerate(genes_shared)}
    ig = np.array([idx_map[g] for g in genes_keep], dtype=int)
    r_case = corr_case[np.ix_(ig, ig)]
    r_ctrl = corr_ctrl[np.ix_(ig, ig)]

    delta = r_case - r_ctrl
    if args.delta_scale == "half":
        delta = delta / 2.0

    rows: list[dict[str, object]] = []
    for p in powers:
        adj_p = adjacency_from_delta(delta, p, args.network_type)
        k = adj_p.sum(axis=1)
        sf = scale_free_fit_from_connectivity(k, bins=int(args.degree_bins))
        rows.append(
            {
                "pair": pair_name,
                "power": int(p),
                "n_genes": int(genes_keep.size),
                "mean_connectivity": float(np.mean(k)),
                "median_connectivity": float(np.median(k)),
                "max_connectivity": float(np.max(k)),
                **sf,
            }
        )

    scan_df = pd.DataFrame(rows).sort_values("power").reset_index(drop=True)
    scan_csv = pair_out / f"{pair_name}_soft_threshold_scan.csv"
    scan_df.to_csv(scan_csv, index=False)

    scan_png = pair_out / f"{pair_name}_soft_threshold_scan.png"
    plot_soft_threshold(scan_df, target_r2=float(args.target_signed_r2), out_png=scan_png, title=f"{pair_name} [{args.network_type}]")

    if args.force_power is not None:
        selected_power = int(args.force_power)
        reason = "forced by --force-power"
    else:
        selected_power, reason = choose_power(
            scan_df,
            target_signed_r2=float(args.target_signed_r2),
            min_mean_k=float(args.min_mean_connectivity),
            r2_plateau_delta=float(args.r2_plateau_delta),
            r2_near_best_delta=float(args.r2_near_best_delta),
        )

    adj = adjacency_from_delta(delta, selected_power, args.network_type)
    adj_file = pair_out / f"{pair_name}_diff_adjacency_beta{selected_power}_{args.network_type}.npz"
    np.savez_compressed(adj_file, adjacency=np.asarray(adj, dtype=np.float32), genes=genes_keep)

    ei, ej, ew = top_edges_from_adjacency(adjacency=adj, min_weight=float(args.candidate_min_weight))

    all_values = pd.read_csv(all_values_file, sep="\t")
    all_values["gene_a"] = all_values["gene_a"].astype(str)
    all_values["gene_b"] = all_values["gene_b"].astype(str)
    all_values_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_values.to_dict(orient="records"):
        row_dict = {str(k): v for k, v in row.items()}
        key = (str(row_dict["gene_a"]), str(row_dict["gene_b"]))
        all_values_lookup[key] = row_dict

    rows_edges: list[dict[str, object]] = []
    for i, j, w in zip(ei, ej, ew):
        ga = str(genes_keep[i])
        gb = str(genes_keep[j])
        rec = all_values_lookup.get((ga, gb))
        if rec is None:
            continue
        rows_edges.append(
            {
                "gene_a": ga,
                "gene_b": gb,
                "weight": float(w),
                "rho_case": float(rec["rho_case"]),
                "rho_control": float(rec["rho_control"]),
                "delta_r": float(rec["rho_case"]) - float(rec["rho_control"]),
                "C": float(rec["C"]),
                "S": float(rec["S"]),
                "D": float(rec["D"]),
            }
        )

    edges_df = pd.DataFrame(rows_edges)
    if not edges_df.empty:
        labels, sel_value = classify_edges_no_threshold(edges_df["C"].to_numpy(), edges_df["S"].to_numpy(), edges_df["D"].to_numpy())
        edges_df["link_type"] = labels
        edges_df["selected_value"] = sel_value
        edges_df = edges_df.sort_values(["selected_value", "weight"], ascending=False).reset_index(drop=True)
        n_edges_pre_topk = int(edges_df.shape[0])
        if int(args.analysis_top_k) > 0:
            edges_df = edges_df.head(int(args.analysis_top_k)).copy().reset_index(drop=True)
    else:
        edges_df = pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "weight",
                "rho_case",
                "rho_control",
                "delta_r",
                "C",
                "S",
                "D",
                "link_type",
                "selected_value",
            ]
        )
        n_edges_pre_topk = 0

    edges_df, wto_edges_df, node_avg_wto = augment_edges_with_wto(edges_df)

    edges_file = pair_out / f"{pair_name}_differential_edges_scalefree.csv"
    wto_edges_file = pair_out / f"{pair_name}_wTO_edges.tsv"
    edges_df.to_csv(edges_file, index=False)
    wto_edges_df.to_csv(wto_edges_file, sep="\t", index=False)

    thresholds = {
        "branch": "scale_free",
        "csd_threshold_mode": "none",
        "k_C": float("nan"),
        "k_S": float("nan"),
        "k_D": float("nan"),
        "selected_power": int(selected_power),
        "selection_reason": reason,
    }
    thresholds_file = pair_out / f"{pair_name}_importance_thresholds.json"
    save_json(thresholds_file, thresholds)

    gene_change_df = pd.read_csv(gene_change_file)
    network_payload = compute_node_and_network_metrics(
        edges_df=edges_df,
        gene_change_df=gene_change_df,
        pair_name=pair_name,
        out_dir=pair_out,
        top_hubs=int(args.top_hubs),
        node_avg_wto=node_avg_wto,
    )
    nodes_df = cast(pd.DataFrame, network_payload["nodes_df"])
    hubs_df = cast(pd.DataFrame, network_payload["top_hubs_df"])
    modules_df = cast(pd.DataFrame, network_payload["modules_df"])
    network_metrics = cast(dict[str, Any], network_payload["network_metrics"])

    nodes_file = pair_out / f"{pair_name}_node_homogeneity_scalefree.csv"
    hubs_file = pair_out / f"{pair_name}_top_hubs_scalefree.csv"
    modules_file = pair_out / f"{pair_name}_louvain_modules.tsv"

    nodes_df.to_csv(nodes_file, index=False)
    hubs_df.to_csv(hubs_file, index=False)
    modules_df.to_csv(modules_file, sep="\t", index=False)

    csd_files = export_csd_files(edges_df=edges_df, out_dir=pair_out, pair_name=pair_name)

    summary = {
        "pair": pair_name,
        "case": case,
        "control": ctrl,
        "branch": "scale_free",
        "network_type": args.network_type,
        "delta_scale": args.delta_scale,
        "selected_power": int(selected_power),
        "selection_reason": reason,
        "candidate_min_weight": float(args.candidate_min_weight),
        "analysis_top_k": int(args.analysis_top_k),
        "n_genes_used": int(genes_keep.size),
        "n_edges_before_topk": int(n_edges_pre_topk),
        "n_edges_exported": int(edges_df.shape[0]),
        "n_nodes_exported": int(network_metrics["n_nodes"]),
        "source_case_corr": str(case_corr_file),
        "source_control_corr": str(ctrl_corr_file),
        "scan_csv": str(scan_csv),
        "scan_png": str(scan_png),
        "adjacency_file": str(adj_file),
        "all_values_file": str(all_values_file),
        "thresholds_file": str(thresholds_file),
        "edges_file": str(edges_file),
        "wto_edges_file": str(wto_edges_file),
        "nodes_file": str(nodes_file),
        "hubs_file": str(hubs_file),
        "modules_file": str(modules_file),
        **network_metrics,
        **csd_files,
    }
    summary_file = pair_out / f"{pair_name}_summary_scalefree.json"
    save_json(summary_file, summary)

    print(f"[{pair_name}] genes={summary['n_genes_used']} beta={selected_power} edges={summary['n_edges_exported']}")
    return summary


def main() -> None:
    args = parse_args()
    corr_root = resolve_base(args.corr_dir)
    shared_root = resolve_base(args.shared_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = list_conditions(corr_root)
    pairs = parse_pairs(args.pair, conditions)
    powers = parse_powers(args.powers)

    summaries: list[dict[str, object]] = []
    for case, ctrl in pairs:
        summaries.append(
            run_pair(
                case=case,
                ctrl=ctrl,
                corr_root=corr_root,
                shared_root=shared_root,
                out_root=out_root,
                powers=powers,
                args=args,
            )
        )

    pd.DataFrame(summaries).sort_values("pair").to_csv(out_root / "differential_scalefree_summary.csv", index=False)
    print(f"Done. Stage-09 branch-A outputs: {out_root}")


if __name__ == "__main__":
    main()
