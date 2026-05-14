#!/usr/bin/env python3
# flake8: noqa: E501
"""Single-condition network builder from permutation thresholds.

Loads the precomputed per-condition Pearson correlation matrix and applies
the permutation threshold from script 23. Gene pairs where |r| >= threshold
are retained as edges. Edge weight is |r|; sign of r is recorded separately.
Basic network metrics are computed and exported.

Inputs:
    results/09_correlation/pearson/<cond>/<cond>_pearson_corr.npz
    results/12_single_condition_thresholds/<cond>/<cond>_permutation_threshold.json

Outputs per condition in results/13_single_condition_networks/<cond>/:
    <cond>_edges.tsv          (gene_a, gene_b, r, abs_r, sign)
    <cond>_nodes.csv          (gene, degree, weighted_degree, clustering_coefficient,
                               closeness_centrality, betweenness_centrality)
    <cond>_top_hubs.csv
    <cond>_network_summary.json
    <cond>_degree_distribution.png
    <cond>_abs_r_distribution.png
    <cond>_hub_subgraph.png
Summary:
    results/13_single_condition_networks/single_condition_networks_summary.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import poisson

from utils.network_utils import load_corr_payload, resolve_base, save_json

CONDITIONS = [
    "ER_tumor",
    "Normal",
    "HER2_tumor",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_BRCA1_tumor",
    "Triple_negative_tumor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-condition network builder")
    parser.add_argument("--corr-dir", default="results/09_correlation/pearson",
                        help="Per-condition correlation output directory")
    parser.add_argument("--threshold-dir", default="results/12_single_condition_thresholds",
                        help="Permutation threshold directory from script 23")
    parser.add_argument("--output-dir", default="results/13_single_condition_networks",
                        help="Output directory for single-condition networks")
    parser.add_argument("--condition", action="append", default=[],
                        help="Condition name to process. Repeat for multiple. Default: all six.")
    parser.add_argument("--top-hubs", type=int, default=50,
                        help="Number of top hub genes to report (by degree)")
    parser.add_argument("--hub-subgraph-n", type=int, default=30,
                        help="Number of top hub genes to include in hub subgraph visualisation")
    return parser.parse_args()


def plot_degree_distribution(g: nx.Graph, out_png: Path, condition: str) -> None:
    degrees = sorted([d for _, d in g.degree()], reverse=True)
    if not degrees:
        return
    mean_deg = float(np.mean(degrees))
    max_deg = max(degrees)
    bins = np.arange(0, max_deg + 2) - 0.5

    # Poisson null: same mean degree
    x_poisson = np.arange(0, max_deg + 1)
    n_nodes = g.number_of_nodes()
    poisson_expected = n_nodes * poisson.pmf(x_poisson, mu=mean_deg)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.hist(degrees, bins=bins, color="#2a9d8f", edgecolor="none", alpha=0.85,
            label="observed")
    ax.plot(x_poisson, poisson_expected, color="#e76f51", linewidth=1.8,
            linestyle="--", label=f"Poisson null (λ = {mean_deg:.1f})")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{condition}: degree distribution")
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_abs_r_distribution(abs_r_vals: np.ndarray, threshold: float,
                             out_png: Path, condition: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.hist(abs_r_vals, bins=60, color="#457b9d", edgecolor="none", alpha=0.85)
    ax.axvline(threshold, linestyle="--", linewidth=1.5, color="#e63946",
               label=f"threshold = {threshold:.4f}")
    ax.set_xlabel("|r|")
    ax.set_ylabel("Number of gene pairs")
    ax.set_title(f"{condition}: |r| distribution of retained edges")
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_hub_subgraph(g: nx.Graph, top_n: int, out_png: Path, condition: str) -> None:
    if g.number_of_nodes() == 0:
        return
    top_genes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_set = {gene for gene, _ in top_genes}
    sub = g.subgraph(top_set)

    degrees = dict(sub.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1200 * (degrees[n] / max_deg) for n in sub.nodes()]
    edge_weights = [sub[u][v].get("weight", 0.5) for u, v in sub.edges()]

    fig, ax = plt.subplots(figsize=(12.0, 10.0), constrained_layout=True)
    pos = nx.spring_layout(sub, seed=42, k=1.8 / np.sqrt(max(sub.number_of_nodes(), 1)))
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.35, width=edge_weights,
                           edge_color="#aaaaaa")
    nc = nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_sizes,
                                node_color=[degrees[n] for n in sub.nodes()],
                                cmap=plt.cm.YlOrRd, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=6.5, font_color="#111111")
    plt.colorbar(nc, ax=ax, label="Degree within top-hub subgraph")
    ax.set_title(f"{condition}: top-{top_n} hub gene subgraph")
    ax.axis("off")
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compute_node_metrics(g: nx.Graph) -> pd.DataFrame:
    degree = dict(g.degree())
    weighted_degree = dict(g.degree(weight="weight"))
    clustering = nx.clustering(g, weight="weight") if g.number_of_nodes() > 0 else {}
    closeness = nx.closeness_centrality(g) if g.number_of_nodes() > 0 else {}

    inv_g = g.copy()
    for u, v, d in inv_g.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["distance"] = 1.0 / max(w, 1e-12)
    betweenness = nx.betweenness_centrality(inv_g, weight="distance") if g.number_of_nodes() > 0 else {}

    return pd.DataFrame({
        "gene": list(degree.keys()),
        "degree": [int(degree[g]) for g in degree],
        "weighted_degree": [float(weighted_degree[g]) for g in degree],
        "clustering_coefficient": [float(clustering.get(g, np.nan)) for g in degree],
        "closeness_centrality": [float(closeness.get(g, np.nan)) for g in degree],
        "betweenness_centrality": [float(betweenness.get(g, np.nan)) for g in degree],
    }).sort_values("degree", ascending=False).reset_index(drop=True)


def run_condition(
    condition: str,
    corr_root: Path,
    threshold_root: Path,
    out_root: Path,
    top_hubs: int,
    hub_subgraph_n: int,
) -> dict[str, object]:
    t0 = time.perf_counter()

    threshold_file = threshold_root / condition / f"{condition}_permutation_threshold.json"
    if not threshold_file.exists():
        raise FileNotFoundError(f"Missing threshold file: {threshold_file}")

    threshold_data = json.loads(threshold_file.read_text())
    threshold = float(threshold_data["threshold_abs_r"])

    corr, genes, corr_path = load_corr_payload(corr_root, condition)

    ii, jj = np.triu_indices(len(genes), k=1)
    r_vals = corr[ii, jj]
    abs_r_vals = np.abs(r_vals)

    mask = abs_r_vals >= threshold
    edge_gene_a = genes[ii[mask]]
    edge_gene_b = genes[jj[mask]]
    edge_r = r_vals[mask]
    edge_abs_r = abs_r_vals[mask]
    edge_sign = np.where(edge_r >= 0, "positive", "negative")

    edges_df = pd.DataFrame({
        "gene_a": edge_gene_a,
        "gene_b": edge_gene_b,
        "r": edge_r,
        "abs_r": edge_abs_r,
        "sign": edge_sign,
    })

    g = nx.Graph()
    for _, row in edges_df.iterrows():
        g.add_edge(str(row["gene_a"]), str(row["gene_b"]), weight=float(row["abs_r"]))

    t_graph = time.perf_counter()
    print(f"  [{condition}] graph built ({t_graph - t0:.1f}s) — "
          f"nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    nodes_df = compute_node_metrics(g)

    t_metrics = time.perf_counter()
    print(f"  [{condition}] node metrics done ({t_metrics - t0:.1f}s)")

    mean_degree = float(np.mean([d for _, d in g.degree()])) if g.number_of_nodes() > 0 else 0.0
    n_pos = int(np.sum(edge_r > 0))
    n_neg = int(np.sum(edge_r < 0))

    cond_out = out_root / condition
    cond_out.mkdir(parents=True, exist_ok=True)

    edges_df.to_csv(cond_out / f"{condition}_edges.tsv", sep="\t", index=False)
    nodes_df.to_csv(cond_out / f"{condition}_nodes.csv", index=False)
    nodes_df.head(top_hubs).to_csv(cond_out / f"{condition}_top_hubs.csv", index=False)

    plot_degree_distribution(g, cond_out / f"{condition}_degree_distribution.png", condition)
    plot_abs_r_distribution(edge_abs_r, threshold,
                             cond_out / f"{condition}_abs_r_distribution.png", condition)
    plot_hub_subgraph(g, hub_subgraph_n,
                      cond_out / f"{condition}_hub_subgraph.png", condition)

    t_plots = time.perf_counter()
    print(f"  [{condition}] plots done ({t_plots - t0:.1f}s)")

    elapsed_total = time.perf_counter() - t0
    summary: dict[str, object] = {
        "condition": condition,
        "threshold_abs_r": threshold,
        "n_genes_input": int(len(genes)),
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_edges_positive": n_pos,
        "n_edges_negative": n_neg,
        "mean_degree": mean_degree,
        "elapsed_seconds": round(elapsed_total, 2),
        "corr_source": str(corr_path),
    }
    save_json(cond_out / f"{condition}_network_summary.json", summary)

    print(
        f"[{condition}] threshold={threshold:.4f} nodes={g.number_of_nodes()} "
        f"edges={g.number_of_edges()} (+{n_pos}/-{n_neg}) elapsed={elapsed_total:.1f}s"
    )
    return summary


def main() -> None:
    args = parse_args()
    corr_root = resolve_base(args.corr_dir)
    threshold_root = resolve_base(args.threshold_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = args.condition if args.condition else CONDITIONS

    t_total = time.perf_counter()
    summaries: list[dict[str, object]] = []
    for condition in conditions:
        summaries.append(run_condition(
            condition=condition,
            corr_root=corr_root,
            threshold_root=threshold_root,
            out_root=out_root,
            top_hubs=args.top_hubs,
            hub_subgraph_n=args.hub_subgraph_n,
        ))

    pd.DataFrame(summaries).to_csv(out_root / "single_condition_networks_summary.csv", index=False)
    total_elapsed = time.perf_counter() - t_total
    print(f"Done. Total elapsed: {total_elapsed:.1f}s. Networks written to: {out_root}")


if __name__ == "__main__":
    main()
