#!/usr/bin/env python3
# flake8: noqa: E501
"""Single-condition co-expression network visualisation.

Reads precomputed edge/node files from script 24 and produces visualisations
in the same style as script 32 (differential network viz):
  - combined PNG: full network (community-aware layout, nodes coloured by
    Louvain module) + top-6 module subgraph panels
  - log-log degree distribution vs Poisson null
  - |r| distribution of retained edges
  - wTO distribution (computed here from the retained-edge graph)

Louvain community detection is applied to the retained-edge weighted graph.
Edge colours encode correlation sign (palette distinct from CSD script 32):
  positive  →  steel blue  #457b9d
  negative  →  purple      #9b5de5

Nodes in components of size < 3 are excluded from the full-network panel
(singletons and pairs that spring physics scatters to the periphery). All
nodes are retained in the analysis, module subgraph panels, and metrics.

wTO is computed from the retained-edge weighted adjacency matrix:
  wTO_ij = (sum_u a_iu * a_uj + a_ij) / (min(k_i, k_j) + 1 - a_ij)

Inputs (from script 24):
  results/13_single_condition_networks/<cond>/<cond>_edges.tsv

Outputs per condition:
  results/13_single_condition_networks/<cond>/<cond>_network_combined.png
  results/13_single_condition_networks/<cond>/<cond>_degree_distribution.png
  results/13_single_condition_networks/<cond>/<cond>_abs_r_distribution.png
  results/13_single_condition_networks/<cond>/<cond>_wto_distribution.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

from utils.network_utils import resolve_base

EDGE_COLORS = {
    "positive": "#f7c948",   # golden yellow — distinct from CSD teal/orange/coral
    "negative": "#9b5de5",   # purple        — distinct from CSD teal/orange/coral
}

_MODULE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d3d3d3", "#888888",
]

MODULE_EDGE_FILTER_THRESHOLD = 500

CONDITIONS = [
    "ER_tumor",
    "Normal",
    "HER2_tumor",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_BRCA1_tumor",
    "Triple_negative_tumor",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-condition network visualisation")
    p.add_argument("--network-dir", default="results/13_single_condition_networks")
    p.add_argument("--condition", action="append", default=[],
                   help="Condition to process. Repeat for multiple. Default: all six.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--top-label-n", type=int, default=15)
    p.add_argument("--viz-edge-cap", type=int, default=15000,
                   help="Max edges shown in full-network panel (by |r|). 0 = all.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# wTO
# ---------------------------------------------------------------------------

def compute_wto(g: nx.Graph) -> dict[tuple[str, str], float]:
    if g.number_of_edges() == 0:
        return {}
    nodes = list(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, d in g.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = float(d.get("weight", 1.0))
        A[i, j] = w
        A[j, i] = w
    k = A.sum(axis=1)
    wto_dict: dict[tuple[str, str], float] = {}
    for u, v, d in g.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        a_ij = float(d.get("weight", 1.0))
        numerator = float(np.dot(A[i], A[j])) + a_ij
        denominator = min(float(k[i]), float(k[j])) + 1.0 - a_ij
        wto = numerator / denominator if denominator > 1e-12 else 0.0
        wto_dict[(u, v)] = float(np.clip(wto, 0.0, 1.0))
    return wto_dict


def run_leiden(g: nx.Graph, seed: int) -> dict[str, int]:
    #returns gene → module_id (1-indexed, largest module = 1)
    if g.number_of_nodes() == 0:
        return {}
    nodes = list(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(
        n=len(nodes),
        edges=[(node_idx[u], node_idx[v]) for u, v in g.edges()],
        edge_attrs={"weight": [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]},
    )
    partition = leidenalg.find_partition(
        ig_g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed
    )
    communities = sorted([list(c) for c in partition], key=len, reverse=True)
    node_to_module: dict[str, int] = {}
    for mod_id, community in enumerate(communities, start=1):
        for i in community:
            node_to_module[str(nodes[i])] = mod_id
    return node_to_module


def modules_sorted_by_size(node_to_module: dict[str, int], g: nx.Graph) -> list[tuple[int, list[str]]]:
    buckets: dict[int, list[str]] = {}
    for node, mod in node_to_module.items():
        if node in g:
            buckets.setdefault(mod, []).append(node)
    return sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)


def module_color(mod_id: int) -> str:
    return _MODULE_PALETTE[(int(mod_id) - 1) % len(_MODULE_PALETTE)]


def community_layout(g: nx.Graph, node_to_module: dict[str, int], seed: int) -> dict[str, tuple[float, float]]:
    mods = modules_sorted_by_size(node_to_module, g)
    n_mods = len(mods)
    if n_mods == 0:
        return nx.spring_layout(g, seed=seed)
    spread = max(15.0, n_mods * 2.5)
    centres: dict[int, tuple[float, float]] = {}
    for i, (mod_id, _) in enumerate(mods):
        angle = 2.0 * math.pi * i / n_mods
        centres[mod_id] = (spread * math.cos(angle), spread * math.sin(angle))
    pos: dict[str, tuple[float, float]] = {}
    for local_idx, (mod_id, nodes) in enumerate(mods):
        subg = g.subgraph(nodes)
        n = len(nodes)
        cx, cy = centres[mod_id]
        if n == 1:
            pos[nodes[0]] = (cx, cy)
        elif n == 2:
            pos[nodes[0]] = (cx - 0.5, cy)
            pos[nodes[1]] = (cx + 0.5, cy)
        else:
            k_param = 2.5 / math.sqrt(n)
            module_scale = max(3.0, math.sqrt(n) * 0.7)
            sub_pos = nx.spring_layout(subg, seed=seed + local_idx, weight="weight",
                                       k=k_param, iterations=60)
            for node, (x, y) in sub_pos.items():
                pos[node] = (cx + x * module_scale, cy + y * module_scale)
    return pos


def filter_edges_for_module_panel(subg: nx.Graph) -> nx.Graph:
    if subg.number_of_edges() <= MODULE_EDGE_FILTER_THRESHOLD:
        return subg
    weights = np.array([d.get("weight", 0.0) for _, _, d in subg.edges(data=True)], dtype=float)
    threshold = float(np.percentile(weights, 75))
    keep_edges = [(u, v) for (u, v, d), w in zip(subg.edges(data=True), weights) if w >= threshold]
    return subg.edge_subgraph(keep_edges).copy()


# ---------------------------------------------------------------------------
# draw helpers
# ---------------------------------------------------------------------------

def _draw_network_on_ax(
    g: nx.Graph,
    pos: dict[str, tuple[float, float]],
    ax: plt.Axes,
    node_to_module: dict[str, int],
    title: str,
    top_label_n: int = 15,
    edge_alpha: float = 0.30,
    node_alpha: float = 0.85,
) -> None:
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    deg_w = dict(g.degree(weight="weight"))
    vals = np.array(list(deg_w.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax > vmin:
        sizes = {n: float(30.0 + 160.0 * ((v - vmin) / (vmax - vmin))) for n, v in deg_w.items()}
    else:
        sizes = {n: 60.0 for n in deg_w}

    node_colors = [module_color(node_to_module.get(n, 1)) for n in g.nodes()]

    pos_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("sign", "positive") == "positive"]
    neg_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("sign") == "negative"]

    if pos_edges:
        nx.draw_networkx_edges(g, pos, edgelist=pos_edges,
                               edge_color=EDGE_COLORS["positive"],
                               width=0.55, alpha=edge_alpha, ax=ax)
    if neg_edges:
        nx.draw_networkx_edges(g, pos, edgelist=neg_edges,
                               edge_color=EDGE_COLORS["negative"],
                               width=0.55, alpha=edge_alpha, ax=ax)

    nx.draw_networkx_nodes(
        g, pos,
        node_size=[sizes.get(n, 40.0) for n in g.nodes()],
        node_color=node_colors, alpha=node_alpha,
        linewidths=0.2, edgecolors="#ffffff", ax=ax,
    )

    top_nodes = sorted(deg_w.items(), key=lambda x: x[1], reverse=True)[:top_label_n]
    labels = {n: n for n, _ in top_nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, font_color="#1d3557", ax=ax)

    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def draw_combined_figure(
    g_full: nx.Graph,
    g_viz: nx.Graph,
    node_to_module: dict[str, int],
    condition: str,
    out_png: Path,
    dpi: int,
    seed: int,
    top_label_n: int,
) -> None:
    cond_label = condition.replace("_", " ")
    mods = modules_sorted_by_size(node_to_module, g_viz)
    n_mods_total = len(mods)

    fig = plt.figure(figsize=(22, 26))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2.5, 1.0, 1.0],
                           hspace=0.10, wspace=0.06, top=0.97, bottom=0.02)
    ax_main = fig.add_subplot(gs[0, :])
    ax_mod_panels = [fig.add_subplot(gs[1, i]) for i in range(3)] + \
                    [fig.add_subplot(gs[2, i]) for i in range(3)]

    # full network — drop components of size < 3 (singletons/pairs scatter to periphery)
    large_component_nodes = {
        n for comp in nx.connected_components(g_viz) if len(comp) >= 3 for n in comp
    }
    g_draw = g_viz.subgraph(large_component_nodes).copy()
    n_excluded = g_viz.number_of_nodes() - g_draw.number_of_nodes()

    spring_k = 1.5 / math.sqrt(max(g_draw.number_of_nodes(), 1))
    comm_pos = nx.spring_layout(g_draw, seed=seed, weight="weight",
                                k=spring_k, iterations=60)

    n_pos = sum(1 for _, _, d in g_draw.edges(data=True) if d.get("sign", "positive") == "positive")
    n_neg = sum(1 for _, _, d in g_draw.edges(data=True) if d.get("sign") == "negative")
    edge_note = ""
    if g_viz.number_of_edges() < g_full.number_of_edges():
        edge_note = f" | top-{g_viz.number_of_edges():,} edges by |r|"
    excluded_note = f" | {n_excluded} singletons/pairs excluded from panel" if n_excluded > 0 else ""

    main_title = (
        f"{cond_label}\n"
        f"nodes={g_full.number_of_nodes():,} | edges={g_full.number_of_edges():,} | "
        f"modules={n_mods_total} | pos={n_pos:,} neg={n_neg:,}{edge_note}{excluded_note}"
    )
    _draw_network_on_ax(g_draw, comm_pos, ax_main, node_to_module,
                        title=main_title, top_label_n=top_label_n, edge_alpha=0.22)

    # edge sign legend
    edge_legend_handles = [
        mlines.Line2D([], [], color=EDGE_COLORS["positive"], linewidth=2.5,
                      label="Positive correlation"),
        mlines.Line2D([], [], color=EDGE_COLORS["negative"], linewidth=2.5,
                      label="Negative correlation"),
    ]
    edge_leg = ax_main.legend(handles=edge_legend_handles, loc="lower left",
                               frameon=True, framealpha=0.88,
                               fontsize=8, title="Edge sign", title_fontsize=8.5)
    ax_main.add_artist(edge_leg)

    # module legend (top-10 by size)
    max_in_legend = min(10, n_mods_total)
    mod_legend_handles = [
        mpatches.Patch(color=module_color(mod_id),
                       label=f"Module {i + 1} — {len(nodes)} nodes")
        for i, (mod_id, nodes) in enumerate(mods[:max_in_legend])
    ]
    if n_mods_total > max_in_legend:
        mod_legend_handles.append(
            mpatches.Patch(color="#aaaaaa",
                           label=f"+ {n_mods_total - max_in_legend} smaller modules")
        )
    ax_main.legend(handles=mod_legend_handles, loc="lower right",
                   frameon=True, framealpha=0.88,
                   fontsize=7.5, title="Modules (ranked by size)", title_fontsize=8.5)

    ax_main.text(0.01, 0.01, "Node size ∝ weighted degree",
                 transform=ax_main.transAxes, fontsize=7, color="#555555",
                 va="bottom", ha="left")

    # top-6 module subgraph panels
    for i, ax in enumerate(ax_mod_panels):
        if i >= len(mods):
            ax.axis("off")
            continue
        mod_id, mod_nodes = mods[i]
        mod_nodes_in_g = [n for n in mod_nodes if n in g_viz]
        raw_subg = g_viz.subgraph(mod_nodes_in_g).copy()
        display_subg = filter_edges_for_module_panel(raw_subg)
        sub_pos = nx.spring_layout(
            display_subg, seed=seed + i + 1, weight="weight",
            k=2.0 / max(1, math.sqrt(len(mod_nodes_in_g))),
        )
        sub_node_to_mod = {n: mod_id for n in display_subg.nodes()}

        n_raw = raw_subg.number_of_edges()
        n_shown = display_subg.number_of_edges()
        edge_note_panel = "" if n_raw == n_shown else f"\n(top-25% |r|, {n_shown} of {n_raw} edges)"

        sub_wdeg = dict(display_subg.degree(weight="weight"))
        top3 = [n for n, _ in sorted(sub_wdeg.items(), key=lambda x: x[1], reverse=True)[:3]]
        hub_note = f"\nTop hubs: {', '.join(top3)}" if top3 else ""
        mod_title = f"Module {i + 1} — {len(mod_nodes_in_g)} nodes{edge_note_panel}{hub_note}"

        _draw_network_on_ax(display_subg, sub_pos, ax, sub_node_to_mod,
                            title=mod_title, top_label_n=8, edge_alpha=0.55)

    fig.suptitle(f"Single-Condition Co-expression Network — {cond_label}",
                 fontsize=11, y=0.99)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_degree_distribution(g: nx.Graph, out_png: Path, condition: str, dpi: int) -> None:
    degrees = np.array([d for _, d in g.degree()], dtype=int)
    if degrees.size == 0:
        return
    k_mean = float(degrees.mean())
    max_k = int(degrees.max())
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    uniq, counts = np.unique(degrees, return_counts=True)
    mask_obs = uniq > 0
    ax.loglog(uniq[mask_obs], counts[mask_obs], "o", color="#1d3557",
              label="Observed", markersize=5)
    k_range = np.arange(1, max_k + 1)
    null_counts = sp_poisson.pmf(k_range, k_mean) * float(g.number_of_nodes())
    mask_null = null_counts > 0.01
    if mask_null.any():
        ax.loglog(k_range[mask_null], null_counts[mask_null],
                  "--", color="#e63946", linewidth=1.5,
                  label=f"Poisson null ($\\bar{{k}}={k_mean:.1f}$)")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"{condition.replace('_', ' ')}: degree distribution", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_abs_r_distribution(edges_df: pd.DataFrame, threshold: float,
                             out_png: Path, condition: str, dpi: int) -> None:
    abs_r = edges_df["abs_r"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    ax.hist(abs_r, bins=60, color="#457b9d", edgecolor="none", alpha=0.85)
    ax.axvline(threshold, linestyle="--", linewidth=1.5, color="#e63946",
               label=f"threshold = {threshold:.4f}")
    ax.set_xlabel("|r|")
    ax.set_ylabel("Number of gene pairs")
    ax.set_title(f"{condition.replace('_', ' ')}: |r| distribution of retained edges", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_wto_distribution(wto_dict: dict[tuple[str, str], float],
                           out_png: Path, condition: str, dpi: int) -> None:
    if not wto_dict:
        return
    wto_vals = np.array(list(wto_dict.values()), dtype=float)
    wto_mean = float(wto_vals.mean())
    wto_median = float(np.median(wto_vals))
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    ax.hist(wto_vals, bins=60, color="#2a9d8f", edgecolor="none", alpha=0.85)
    ax.axvline(wto_mean, color="#e76f51", linewidth=1.5, linestyle="--",
               label=f"Mean {wto_mean:.3f}")
    ax.axvline(wto_median, color="#f4a261", linewidth=1.5, linestyle=":",
               label=f"Median {wto_median:.3f}")
    ax.set_xlabel("wTO")
    ax.set_ylabel("Number of edges")
    ax.set_title(f"{condition.replace('_', ' ')}: edge wTO distribution", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# per-condition runner
# ---------------------------------------------------------------------------

def run_condition(condition: str, net_root: Path, args: argparse.Namespace) -> None:
    import json

    cond_dir = net_root / condition
    edges_file = cond_dir / f"{condition}_edges.tsv"
    if not edges_file.exists():
        print(f"  [{condition}] SKIP — missing {edges_file}")
        return

    edges_df = pd.read_csv(edges_file, sep="\t")
    if edges_df.empty:
        print(f"  [{condition}] SKIP — empty edge file")
        return
    print(f"  [{condition}] {len(edges_df):,} edges")

    # full graph for wTO and stats
    g_full = nx.Graph()
    for _, row in edges_df.iterrows():
        g_full.add_edge(str(row["gene_a"]), str(row["gene_b"]),
                        weight=float(row["abs_r"]), sign=str(row["sign"]), r=float(row["r"]))

    # visualisation graph — subsample if too dense
    if args.viz_edge_cap > 0 and len(edges_df) > args.viz_edge_cap:
        edges_viz_df = edges_df.nlargest(args.viz_edge_cap, "abs_r").reset_index(drop=True)
    else:
        edges_viz_df = edges_df

    g_viz = nx.Graph()
    for _, row in edges_viz_df.iterrows():
        g_viz.add_edge(str(row["gene_a"]), str(row["gene_b"]),
                       weight=float(row["abs_r"]), sign=str(row["sign"]), r=float(row["r"]))

    print(f"  [{condition}] Leiden community detection ...")
    node_to_module = run_leiden(g_full, seed=args.seed)
    n_mods = len(set(node_to_module.values()))
    print(f"  [{condition}] {n_mods} modules")

    # wTO on full graph
    print(f"  [{condition}] computing wTO …")
    wto_dict = compute_wto(g_full)

    # threshold
    threshold_file = resolve_base("results/12_single_condition_thresholds") / condition / f"{condition}_permutation_threshold.json"
    threshold = float("nan")
    if threshold_file.exists():
        threshold = float(json.loads(threshold_file.read_text())["threshold_abs_r"])

    print(f"  [{condition}] drawing …")
    draw_combined_figure(
        g_full=g_full, g_viz=g_viz, node_to_module=node_to_module,
        condition=condition,
        out_png=cond_dir / f"{condition}_network_combined.png",
        dpi=args.dpi, seed=args.seed, top_label_n=args.top_label_n,
    )
    draw_degree_distribution(g_full, cond_dir / f"{condition}_degree_distribution.png",
                             condition, args.dpi)
    if not math.isnan(threshold):
        draw_abs_r_distribution(edges_df, threshold,
                                cond_dir / f"{condition}_abs_r_distribution.png",
                                condition, args.dpi)
    draw_wto_distribution(wto_dict, cond_dir / f"{condition}_wto_distribution.png",
                          condition, args.dpi)

    print(f"  [{condition}] done")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    net_root = resolve_base(args.network_dir)
    conditions = args.condition if args.condition else CONDITIONS
    for condition in conditions:
        print(f"\n[{condition}]")
        run_condition(condition, net_root, args)


if __name__ == "__main__":
    main()
