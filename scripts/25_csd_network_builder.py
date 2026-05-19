#!/usr/bin/env python3
# flake8: noqa: E501
"""Build CSD differential networks from permutation-derived C/S/D thresholds."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from utils.network_utils import (
    augment_edges_with_wto,
    classify_edges_no_threshold,
    compute_node_and_network_metrics,
    export_csd_files,
    resolve_base,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build permutation-thresholded CSD networks")
    parser.add_argument(
        "--shared-dir",
        default="results/12_csd_scoring",
        help="CSD scoring output directory from 23_csd_scoring.py",
    )
    parser.add_argument(
        "--threshold-dir",
        default="results/13_csd_thresholds",
        help="Permutation threshold output directory from 24_csd_permutation_thresholds.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/14_csd_networks",
        help="Output directory for built CSD networks",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Pair folder name case__vs__control. Repeat for multiple",
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
        help="Optional analysis cap after permutation thresholding (0 keeps all)",
    )
    return parser.parse_args()


def run_pair(
    pair_dir: Path,
    threshold_root: Path,
    out_root: Path,
    top_hubs: int,
    analysis_top_k: int,
) -> dict[str, object]:
    t0 = time.perf_counter()
    pair_name = pair_dir.name
    all_values_file = pair_dir / f"{pair_name}_AllValues.tsv"
    gene_change_file = pair_dir / f"{pair_name}_gene_expression_change.csv"
    threshold_file = threshold_root / pair_name / f"{pair_name}_permutation_thresholds.json"
    presence_file = pair_dir / f"{pair_name}_gene_presence.csv"

    if not all_values_file.exists() or not gene_change_file.exists() or not threshold_file.exists():
        raise FileNotFoundError(f"Missing required files for {pair_name}")

    #load condition-specific genes (present in only one condition)
    condition_specific_genes: set[str] = set()
    if presence_file.exists():
        pres_df = pd.read_csv(presence_file)
        condition_specific_genes = set(pres_df.loc[pres_df["presence"] != "shared", "gene"].astype(str).tolist())

    #load allvalues and apply permutation thresholds to select edges
    all_values = pd.read_csv(all_values_file, sep="\t")
    thresholds = pd.read_json(threshold_file, typ="series")

    t_c = float(thresholds["threshold_C"])
    t_s = float(thresholds["threshold_S"])
    t_d = float(thresholds["threshold_D"])

    keep = (all_values["C"] > t_c) | (all_values["S"] > t_s) | (all_values["D"] > t_d)
    edges_df = all_values.loc[keep].copy().reset_index(drop=True)

    if not edges_df.empty:
        #classify edges and relabel condition-specific C edges as S
        #(gene absent in one condition fills rho_other=0, making C==S; argmax picks C but semantically these are S)
        labels, sel_value = classify_edges_no_threshold(
            edges_df["C"].to_numpy(), edges_df["S"].to_numpy(), edges_df["D"].to_numpy())
        edges_df["link_type"] = labels
        edges_df["selected_value"] = sel_value
        if condition_specific_genes:
            csg_mask = (
                (edges_df["link_type"] == "C")
                & (edges_df["gene_a"].isin(condition_specific_genes) | edges_df["gene_b"].isin(condition_specific_genes))
            )
            edges_df.loc[csg_mask, "link_type"] = "S"
            #s col equals c col for these pairs since c==s when rho_other=0
            edges_df.loc[csg_mask, "selected_value"] = edges_df.loc[csg_mask, "S"]
        edges_df["weight"] = edges_df["selected_value"]
        edges_df["delta_r"] = edges_df["rho_case"] - edges_df["rho_control"]
        edges_df = edges_df[
            ["gene_a", "gene_b", "weight", "rho_case", "rho_control", "delta_r", "C", "S", "D", "link_type", "selected_value"]
        ].sort_values(["selected_value", "weight"], ascending=False).reset_index(drop=True)
        n_edges_pre_topk = int(edges_df.shape[0])
        if int(analysis_top_k) > 0:
            edges_df = edges_df.head(int(analysis_top_k)).copy().reset_index(drop=True)
    else:
        edges_df = pd.DataFrame(
            columns=["gene_a", "gene_b", "weight", "rho_case", "rho_control", "delta_r", "C", "S", "D", "link_type", "selected_value"]
        )
        n_edges_pre_topk = 0

    #augment with wto overlap scores
    edges_df, wto_edges_df, node_avg_wto = augment_edges_with_wto(edges_df)

    #save edge tables
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    edges_file = pair_out / f"{pair_name}_differential_edges_permutation.csv"
    wto_edges_file = pair_out / f"{pair_name}_wTO_edges.tsv"
    edges_df.to_csv(edges_file, index=False)
    wto_edges_df.to_csv(wto_edges_file, sep="\t", index=False)

    #compute node topology metrics, hubs, and leiden modules
    gene_change_df = pd.read_csv(gene_change_file)
    network_payload = compute_node_and_network_metrics(
        edges_df=edges_df,
        gene_change_df=gene_change_df,
        pair_name=pair_name,
        out_dir=pair_out,
        top_hubs=int(top_hubs),
        node_avg_wto=node_avg_wto,
    )
    nodes_df = cast(pd.DataFrame, network_payload["nodes_df"])
    hubs_df = cast(pd.DataFrame, network_payload["top_hubs_df"])
    modules_df = cast(pd.DataFrame, network_payload["modules_df"])
    network_metrics = cast(dict[str, Any], network_payload["network_metrics"])

    #save node, hub, module, and csd-split edge files
    nodes_file = pair_out / f"{pair_name}_node_homogeneity_permutation.csv"
    hubs_file = pair_out / f"{pair_name}_top_hubs_permutation.csv"
    modules_file = pair_out / f"{pair_name}_leiden_modules.tsv"

    nodes_df.to_csv(nodes_file, index=False)
    hubs_df.to_csv(hubs_file, index=False)
    modules_df.to_csv(modules_file, sep="\t", index=False)

    csd_files = export_csd_files(edges_df=edges_df, out_dir=pair_out, pair_name=pair_name)

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "pair": pair_name,
        "branch": "permutation_union",
        "n_condition_specific_genes": int(len(condition_specific_genes)),
        "threshold_C": t_c,
        "threshold_S": t_s,
        "threshold_D": t_d,
        "analysis_top_k": int(analysis_top_k),
        "n_edges_before_topk": int(n_edges_pre_topk),
        "n_edges_exported": int(edges_df.shape[0]),
        "n_nodes_exported": int(network_metrics["n_nodes"]),
        "elapsed_seconds": elapsed,
        "threshold_file": str(threshold_file),
        "all_values_file": str(all_values_file),
        "edges_file": str(edges_file),
        "wto_edges_file": str(wto_edges_file),
        "nodes_file": str(nodes_file),
        "hubs_file": str(hubs_file),
        "modules_file": str(modules_file),
        **network_metrics,
        **csd_files,
    }

    summary_file = pair_out / f"{pair_name}_summary_permutation.json"
    save_json(summary_file, summary)

    print(
        f"[{pair_name}] thresholds=({t_c:.4f},{t_s:.4f},{t_d:.4f}) "
        f"edges={summary['n_edges_exported']} elapsed={elapsed}s"
    )
    return summary


def main() -> int:
    args = parse_args()

    #resolve paths and create output dir
    shared_root = resolve_base(args.shared_dir)
    threshold_root = resolve_base(args.threshold_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    #discover pair dirs and filter to requested subset if specified
    pair_dirs = sorted([p for p in shared_root.iterdir() if p.is_dir() and "__vs__" in p.name])
    if args.pair:
        keep = set(args.pair)
        pair_dirs = [p for p in pair_dirs if p.name in keep]

    if not pair_dirs:
        raise ValueError("No pair directories found to process")

    #build network for each pair
    summaries: list[dict[str, object]] = []
    for pair_dir in pair_dirs:
        summaries.append(run_pair(
            pair_dir=pair_dir, threshold_root=threshold_root, out_root=out_root,
            top_hubs=int(args.top_hubs), analysis_top_k=int(args.analysis_top_k),
        ))

    #write combined summary csv
    pd.DataFrame(summaries).sort_values("pair").to_csv(out_root / "differential_permutation_summary.csv", index=False)
    total = round(sum(s.get("elapsed_seconds", 0) for s in summaries), 2)
    print(f"Done. Total elapsed: {total}s. Outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
