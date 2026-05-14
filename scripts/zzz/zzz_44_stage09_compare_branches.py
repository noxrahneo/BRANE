#!/usr/bin/env python3
# flake8: noqa: E501
"""Stage-09 comparison layer between branch A and branch B outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.network_utils import resolve_base, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stage-09 branch A vs branch B")
    parser.add_argument(
        "--branch-a-dir",
        default="results/15_scalefree_networks",
        help="Branch A output directory",
    )
    parser.add_argument(
        "--branch-b-dir",
        default="results/17_permutation_networks",
        help="Branch B output directory",
    )
    parser.add_argument(
        "--output-dir",
        default="results/09_differential_restructured/05_comparison",
        help="Comparison output directory",
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
        help="Top hubs used for overlap comparison",
    )
    return parser.parse_args()


def edge_key_df(df: pd.DataFrame) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for a, b in zip(df["gene_a"].astype(str), df["gene_b"].astype(str)):
        keys.add((a, b) if a <= b else (b, a))
    return keys


def type_props(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "link_type" not in df.columns:
        return {"C": float("nan"), "S": float("nan"), "D": float("nan")}
    vc = df["link_type"].value_counts(normalize=True)
    return {
        "C": float(vc.get("C", 0.0)),
        "S": float(vc.get("S", 0.0)),
        "D": float(vc.get("D", 0.0)),
    }


def load_summary_metrics(path: Path) -> dict[str, float]:
    s = pd.read_json(path, typ="series")
    out: dict[str, float] = {}
    for k in [
        "assortativity_coefficient",
        "global_modularity_q",
        "average_clustering_coefficient",
        "knn_exponent",
        "n_nodes",
        "n_edges",
    ]:
        out[k] = float(s.get(k, np.nan))
    return out


def run_pair(pair_name: str, a_root: Path, b_root: Path, out_root: Path, top_hubs: int) -> dict[str, object]:
    pair_a = a_root / pair_name
    pair_b = b_root / pair_name

    a_edges_file = pair_a / f"{pair_name}_differential_edges_scalefree.csv"
    b_edges_file = pair_b / f"{pair_name}_differential_edges_permutation.csv"
    a_hubs_file = pair_a / f"{pair_name}_top_hubs_scalefree.csv"
    b_hubs_file = pair_b / f"{pair_name}_top_hubs_permutation.csv"
    a_summary_file = pair_a / f"{pair_name}_summary_scalefree.json"
    b_summary_file = pair_b / f"{pair_name}_summary_permutation.json"

    for req in [a_edges_file, b_edges_file, a_hubs_file, b_hubs_file, a_summary_file, b_summary_file]:
        if not req.exists():
            raise FileNotFoundError(f"Missing comparison input file: {req}")

    a_edges = pd.read_csv(a_edges_file)
    b_edges = pd.read_csv(b_edges_file)

    keys_a = edge_key_df(a_edges)
    keys_b = edge_key_df(b_edges)
    shared = keys_a.intersection(keys_b)

    n_a = len(keys_a)
    n_b = len(keys_b)
    n_shared = len(shared)
    jaccard = float(n_shared / max(len(keys_a.union(keys_b)), 1))

    props_a = type_props(a_edges)
    props_b = type_props(b_edges)

    a_hubs = pd.read_csv(a_hubs_file).head(int(top_hubs))
    b_hubs = pd.read_csv(b_hubs_file).head(int(top_hubs))
    hubs_a = set(a_hubs["gene"].astype(str).tolist()) if not a_hubs.empty else set()
    hubs_b = set(b_hubs["gene"].astype(str).tolist()) if not b_hubs.empty else set()

    hubs_shared = sorted(hubs_a.intersection(hubs_b))
    hubs_a_only = sorted(hubs_a - hubs_b)
    hubs_b_only = sorted(hubs_b - hubs_a)

    metrics_a = load_summary_metrics(a_summary_file)
    metrics_b = load_summary_metrics(b_summary_file)

    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "pair": pair_name,
                "n_edges_branch_a": n_a,
                "n_edges_branch_b": n_b,
                "n_edges_shared": n_shared,
                "edge_jaccard": jaccard,
                "prop_C_a": props_a["C"],
                "prop_S_a": props_a["S"],
                "prop_D_a": props_a["D"],
                "prop_C_b": props_b["C"],
                "prop_S_b": props_b["S"],
                "prop_D_b": props_b["D"],
            }
        ]
    ).to_csv(pair_out / "comparison_edge_overlap.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": pair_name,
                "top_hubs_k": int(top_hubs),
                "n_hubs_a": int(len(hubs_a)),
                "n_hubs_b": int(len(hubs_b)),
                "n_hubs_shared": int(len(hubs_shared)),
                "hubs_shared": ";".join(hubs_shared),
                "hubs_a_only": ";".join(hubs_a_only),
                "hubs_b_only": ";".join(hubs_b_only),
            }
        ]
    ).to_csv(pair_out / "comparison_hub_overlap.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": pair_name,
                "assortativity_a": metrics_a["assortativity_coefficient"],
                "assortativity_b": metrics_b["assortativity_coefficient"],
                "modularity_q_a": metrics_a["global_modularity_q"],
                "modularity_q_b": metrics_b["global_modularity_q"],
                "avg_clustering_a": metrics_a["average_clustering_coefficient"],
                "avg_clustering_b": metrics_b["average_clustering_coefficient"],
                "knn_exponent_a": metrics_a["knn_exponent"],
                "knn_exponent_b": metrics_b["knn_exponent"],
                "n_nodes_a": metrics_a["n_nodes"],
                "n_nodes_b": metrics_b["n_nodes"],
                "n_edges_a": metrics_a["n_edges"],
                "n_edges_b": metrics_b["n_edges"],
            }
        ]
    ).to_csv(pair_out / "comparison_network_metrics.csv", index=False)

    summary = {
        "pair": pair_name,
        "edge_overlap": {
            "n_edges_branch_a": n_a,
            "n_edges_branch_b": n_b,
            "n_edges_shared": n_shared,
            "edge_jaccard": jaccard,
            "csd_type_props_a": props_a,
            "csd_type_props_b": props_b,
        },
        "hub_overlap": {
            "top_hubs_k": int(top_hubs),
            "n_hubs_shared": int(len(hubs_shared)),
            "hubs_shared": hubs_shared,
            "hubs_a_only": hubs_a_only,
            "hubs_b_only": hubs_b_only,
        },
        "network_metrics": {
            "branch_a": metrics_a,
            "branch_b": metrics_b,
        },
    }

    save_json(pair_out / "comparison_summary.json", summary)

    return {
        "pair": pair_name,
        "n_edges_branch_a": n_a,
        "n_edges_branch_b": n_b,
        "n_edges_shared": n_shared,
        "edge_jaccard": jaccard,
        "n_hubs_shared": int(len(hubs_shared)),
        "assortativity_a": metrics_a["assortativity_coefficient"],
        "assortativity_b": metrics_b["assortativity_coefficient"],
        "modularity_q_a": metrics_a["global_modularity_q"],
        "modularity_q_b": metrics_b["global_modularity_q"],
    }


def main() -> None:
    args = parse_args()
    a_root = resolve_base(args.branch_a_dir)
    b_root = resolve_base(args.branch_b_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs_a = {p.name for p in a_root.iterdir() if p.is_dir() and "__vs__" in p.name}
    pairs_b = {p.name for p in b_root.iterdir() if p.is_dir() and "__vs__" in p.name}
    pairs = sorted(pairs_a.intersection(pairs_b))

    if args.pair:
        keep = set(args.pair)
        pairs = [p for p in pairs if p in keep]

    if not pairs:
        raise ValueError("No overlapping pair folders found between branch A and B")

    rows: list[dict[str, object]] = []
    for pair_name in pairs:
        rows.append(run_pair(pair_name, a_root, b_root, out_root, int(args.top_hubs)))

    edge_df = pd.DataFrame(rows).sort_values("pair")
    edge_df[["pair", "n_edges_branch_a", "n_edges_branch_b", "n_edges_shared", "edge_jaccard"]].to_csv(
        out_root / "comparison_edge_overlap.csv", index=False
    )
    edge_df[["pair", "n_hubs_shared"]].to_csv(out_root / "comparison_hub_overlap.csv", index=False)
    edge_df[
        [
            "pair",
            "assortativity_a",
            "assortativity_b",
            "modularity_q_a",
            "modularity_q_b",
        ]
    ].to_csv(out_root / "comparison_network_metrics.csv", index=False)

    summary = {
        "n_pairs": int(len(rows)),
        "pairs": [r["pair"] for r in rows],
        "mean_edge_jaccard": float(edge_df["edge_jaccard"].mean()),
        "mean_shared_hubs": float(edge_df["n_hubs_shared"].mean()),
    }
    save_json(out_root / "comparison_summary.json", summary)

    print(f"Done. Stage-09 comparison outputs: {out_root}")


if __name__ == "__main__":
    main()
