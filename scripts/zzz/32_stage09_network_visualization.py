#!/usr/bin/env python3
# flake8: noqa: E501
"""Stage-09 branch B network visualisations.

Produces per-pair:
  - combined PNG: full network (community layout) + top-3 module subgraphs
  - interactive HTML (nodes coloured by Louvain module, edges by CSD type)
  - degree distribution PNG with Poisson null overlay
  - wTO edge weight distribution PNG

Community assignments are read from script 31 output (_leiden_modules.tsv).
Edge colours: C=#2a9d8f, S=#f4a261, D=#e76f51 (consistent with scripts 30/31).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

try:
    from pyvis.network import Network  # type: ignore[import-not-found]
    HAS_PYVIS = True
except Exception:
    Network = None  # type: ignore[assignment]
    HAS_PYVIS = False

from utils.network_utils import resolve_base

# CSD edge colours — changed: Conserved=C (blue), Specific=S (green), Differentiated=D (red)
# Updated to: C -> blue, S -> green, D -> red
EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}

# qualitative module colour palette (up to 12 modules; wraps for more)
_MODULE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d3d3d3", "#888888",
]

# module panel edge cap: filter to top-25% wTO within the subgraph when edge count exceeds this
MODULE_EDGE_FILTER_THRESHOLD = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Branch B network visualisations")
    p.add_argument("--branch-b-dir", default="results/16_permutation_networks", help="Branch B network outputs from script 31")
    p.add_argument("--output-dir", default="results/18_differential_viz")
    p.add_argument("--pair", action="append", default=[], help="Pair folder name case__vs__control. Repeat for multiple.")
    p.add_argument("--viz-top-k", type=int, default=0, help="Visualisation edge cap (0 = all)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--interactive-html", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top-label-n", type=int, default=15, help="Hub genes to label in full-network panel")
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def module_color(mod_id: int) -> str:
    return _MODULE_PALETTE[(int(mod_id) - 1) % len(_MODULE_PALETTE)]


def read_leiden_modules(modules_file: Path) -> dict[str, int]:
    df = pd.read_csv(modules_file, sep="\t")
    return {str(row["gene"]): int(row["module"]) for _, row in df.iterrows()}


def modules_sorted_by_size(node_to_module: dict[str, int], g: nx.Graph) -> list[tuple[int, list[str]]]:
    """Return list of (module_id, [genes]) sorted by node count descending, restricted to nodes in g."""
    buckets: dict[int, list[str]] = {}
    for node, mod in node_to_module.items():
        if node in g:
            buckets.setdefault(mod, []).append(node)
    return sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)


def build_graph(edges_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    if edges_df.empty:
        return g
    for row in edges_df.to_dict(orient="records"):
        g.add_edge(
            str(row["gene_a"]),
            str(row["gene_b"]),
            weight=float(row.get("weight", 1.0)),
            link_type=str(row.get("link_type", "C")),
            wTO=float(row.get("wTO", float("nan"))),
        )
    return g


def community_layout(
    g: nx.Graph,
    node_to_module: dict[str, int],
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """Hierarchical layout: spring physics within each community, communities spread on a circle."""
    mods = modules_sorted_by_size(node_to_module, g)
    n_mods = len(mods)
    if n_mods == 0:
        return nx.spring_layout(g, seed=seed)

    # place community centres on a circle; radius scales with number of modules
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
            # k controls node spacing within the community; scale proportional to sqrt(n)
            k_param = 2.5 / math.sqrt(n)
            module_scale = max(3.0, math.sqrt(n) * 0.7)
            sub_pos = nx.spring_layout(subg, seed=seed + local_idx, weight="weight", k=k_param, iterations=60)
            for node, (x, y) in sub_pos.items():
                pos[node] = (cx + x * module_scale, cy + y * module_scale)

    return pos


def filter_edges_for_module_panel(subg: nx.Graph) -> nx.Graph:
    """If subgraph is dense, keep only top-25% edges by wTO. Preserves intra-module structure."""
    if subg.number_of_edges() <= MODULE_EDGE_FILTER_THRESHOLD:
        return subg
    wto_vals = [d.get("wTO", float("nan")) for _, _, d in subg.edges(data=True)]
    wto_arr = np.array(wto_vals, dtype=float)
    finite = wto_arr[np.isfinite(wto_arr)]
    if finite.size == 0:
        # no wTO available — keep top-25% by weight
        weights = np.array([d.get("weight", 0.0) for _, _, d in subg.edges(data=True)], dtype=float)
        threshold = float(np.percentile(weights, 75))
        keep_edges = [(u, v) for (u, v, d), w in zip(subg.edges(data=True), weights) if w >= threshold]
    else:
        threshold = float(np.percentile(finite, 75))
        keep_edges = [(u, v) for u, v, d in subg.edges(data=True) if float(d.get("wTO", float("nan"))) >= threshold]
    return subg.edge_subgraph(keep_edges).copy()


# ---------------------------------------------------------------------------
# draw functions
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

    node_colors = [module_color(node_to_module.get(n, 0)) for n in g.nodes()]

    edge_groups: dict[str, list[tuple[Any, Any]]] = {"C": [], "S": [], "D": [], "other": []}
    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "other"))
        edge_groups[t if t in EDGE_COLORS else "other"].append((u, v))

    for t, edges in edge_groups.items():
        if edges:
            nx.draw_networkx_edges(
                g, pos, edgelist=edges,
                edge_color=EDGE_COLORS.get(t, "#8d99ae"),
                width=0.55, alpha=edge_alpha, ax=ax,
            )

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
    g: nx.Graph,
    pair_name: str,
    node_to_module: dict[str, int],
    out_png: Path,
    dpi: int,
    seed: int,
    top_label_n: int = 15,
) -> None:
    """Full network (community layout) on top + top-6 module subgraphs on 2×3 grid below."""
    mods = modules_sorted_by_size(node_to_module, g)
    n_mods_total = len(mods)

    fig = plt.figure(figsize=(22, 26))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2.5, 1.0, 1.0], hspace=0.10, wspace=0.06, top=0.97, bottom=0.02)
    ax_main = fig.add_subplot(gs[0, :])
    ax_mod_panels = [fig.add_subplot(gs[1, i]) for i in range(3)] + [fig.add_subplot(gs[2, i]) for i in range(3)]

    # full network — community layout
    comm_pos = community_layout(g, node_to_module, seed=seed)
    main_title = (
        f"{pair_name.replace('__vs__', ' vs ').replace('_', ' ')}\n"
        f"nodes={g.number_of_nodes()} | edges={g.number_of_edges()} | modules={n_mods_total}"
    )
    _draw_network_on_ax(
        g, comm_pos, ax_main, node_to_module,
        title=main_title, top_label_n=top_label_n, edge_alpha=0.22,
    )

    # edge type legend — lower left of main panel
    edge_legend_handles = [
        mlines.Line2D([], [], color=EDGE_COLORS["C"], linewidth=2.5, label="C — conserved direction"),
        mlines.Line2D([], [], color=EDGE_COLORS["S"], linewidth=2.5, label="S — condition-specific"),
        mlines.Line2D([], [], color=EDGE_COLORS["D"], linewidth=2.5, label="D — direction reversal"),
    ]
    edge_leg = ax_main.legend(
        handles=edge_legend_handles,
        loc="lower left", frameon=True, framealpha=0.88,
        fontsize=8, title="Edge type", title_fontsize=8.5,
    )
    ax_main.add_artist(edge_leg)  # keep it when adding module legend below

    # module colour legend — lower right of main panel (top-10 by size, note if more)
    max_in_legend = min(10, n_mods_total)
    mod_legend_handles = [
        mpatches.Patch(color=module_color(mod_id), label=f"Module {i + 1} — {len(nodes)} nodes")
        for i, (mod_id, nodes) in enumerate(mods[:max_in_legend])
    ]
    if n_mods_total > max_in_legend:
        mod_legend_handles.append(
            mpatches.Patch(color="#aaaaaa", label=f"+ {n_mods_total - max_in_legend} smaller modules")
        )
    ax_main.legend(
        handles=mod_legend_handles,
        loc="lower right", frameon=True, framealpha=0.88,
        fontsize=7.5, title="Modules (ranked by size)", title_fontsize=8.5,
    )

    # node size note
    ax_main.text(
        0.01, 0.01, "Node size ∝ weighted degree",
        transform=ax_main.transAxes, fontsize=7, color="#555555",
        va="bottom", ha="left",
    )

    # top-3 module subgraphs — intra-module edges only, filtered by wTO if dense
    for i, ax in enumerate(ax_mod_panels):
        if i >= len(mods):
            ax.axis("off")
            continue
        mod_id, mod_nodes = mods[i]
        mod_nodes_in_g = [n for n in mod_nodes if n in g]
        raw_subg = g.subgraph(mod_nodes_in_g).copy()
        display_subg = filter_edges_for_module_panel(raw_subg)
        sub_pos = nx.spring_layout(display_subg, seed=seed + i + 1, weight="weight", k=2.0 / max(1, math.sqrt(len(mod_nodes_in_g))))
        sub_node_to_mod = {n: mod_id for n in display_subg.nodes()}

        n_raw_edges = raw_subg.number_of_edges()
        n_shown_edges = display_subg.number_of_edges()
        edge_note = "" if n_raw_edges == n_shown_edges else f"\n(top-25% wTO, {n_shown_edges} of {n_raw_edges} edges shown)"
        # Top 3 hubs by weighted degree within this module subgraph.
        sub_wdeg = dict(display_subg.degree(weight="weight"))
        top3 = [n for n, _ in sorted(sub_wdeg.items(), key=lambda x: x[1], reverse=True)[:3]]
        hub_note = f"\nTop hubs: {', '.join(top3)}" if top3 else ""
        mod_title = f"Module {i + 1} — {len(mod_nodes_in_g)} nodes{edge_note}{hub_note}"

        _draw_network_on_ax(
            display_subg, sub_pos, ax, sub_node_to_mod,
            title=mod_title, top_label_n=8, edge_alpha=0.55,
        )

    fig.suptitle("Differential Co-expression Network — Branch B Permutation", fontsize=11, y=0.99)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_html(
    g: nx.Graph,
    out_html: Path,
    title: str,
    node_to_module: dict[str, int],
) -> None:
    if not HAS_PYVIS or Network is None:
        return
    net: Any = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#1f2937")
    net.barnes_hut(gravity=-12000, central_gravity=0.2, spring_length=170, spring_strength=0.02)

    deg_w = dict(g.degree(weight="weight"))
    for n in g.nodes():
        d_w = float(deg_w.get(n, 0.0))
        mod = node_to_module.get(n, 0)
        net.add_node(
            n, label=n,
            title=f"gene: {n}<br>weighted_degree: {d_w:.3f}<br>module: {mod}",
            color=module_color(mod),
            size=float(8.0 + min(35.0, math.sqrt(max(d_w, 0.0)) * 2.4)),
        )

    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "C"))
        w = float(d.get("weight", 1.0))
        wto = d.get("wTO", float("nan"))
        edge_title = f"type={t}<br>weight={w:.4f}"
        if isinstance(wto, float) and math.isfinite(wto):
            edge_title += f"<br>wTO={wto:.4f}"
        net.add_edge(u, v, value=max(0.1, w), color=EDGE_COLORS.get(t, "#8d99ae"), title=edge_title)

    net.set_options("""
        var options = {
          "physics": {"enabled": true, "stabilization": {"iterations": 300}},
          "interaction": {"hover": true, "navigationButtons": true},
          "nodes": {"font": {"size": 13}}
        }
    """)
    net.write_html(str(out_html), open_browser=False, notebook=False)


def draw_degree_distribution(
    g: nx.Graph,
    out_png: Path,
    pair_name: str,
    dpi: int,
) -> None:
    degrees = np.array([d for _, d in g.degree()], dtype=int)
    if degrees.size == 0:
        return

    k_mean = float(degrees.mean())
    max_k = int(degrees.max())

    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)

    uniq, counts = np.unique(degrees, return_counts=True)
    mask_obs = uniq > 0
    ax.loglog(uniq[mask_obs], counts[mask_obs], "o", color="#1d3557", label="Observed", markersize=5)

    # Poisson null — expected count per degree value under Erdős–Rényi
    k_range = np.arange(1, max_k + 1)
    null_counts = sp_poisson.pmf(k_range, k_mean) * float(g.number_of_nodes())
    mask_null = null_counts > 0.01
    if mask_null.any():
        ax.loglog(
            k_range[mask_null], null_counts[mask_null],
            "--", color="#e63946", linewidth=1.5,
            label=f"Poisson null ($\\bar{{k}}={k_mean:.1f}$)",
        )

    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"{pair_name.replace('__vs__', ' vs ').replace('_', ' ')}: degree distribution", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_wto_distribution(
    edges_df: pd.DataFrame,
    out_png: Path,
    pair_name: str,
    dpi: int,
) -> None:
    wto = pd.to_numeric(edges_df.get("wTO", pd.Series(dtype=float)), errors="coerce").dropna()
    if wto.empty:
        return

    wto_mean = float(wto.mean())
    wto_median = float(wto.median())

    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    ax.hist(wto.to_numpy(), bins=60, color="#2a9d8f", edgecolor="none", alpha=0.85)
    ax.axvline(wto_mean, color="#e76f51", linewidth=1.5, linestyle="--", label=f"Mean {wto_mean:.3f}")
    ax.axvline(wto_median, color="#f4a261", linewidth=1.5, linestyle=":", label=f"Median {wto_median:.3f}")
    ax.set_xlabel("wTO")
    ax.set_ylabel("Number of edges")
    ax.set_title(f"{pair_name.replace('__vs__', ' vs ').replace('_', ' ')}: edge wTO distribution", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# per-pair runner
# ---------------------------------------------------------------------------

def run_pair(
    pair_name: str,
    branch_b_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    edges_file = branch_b_dir / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    modules_file = branch_b_dir / pair_name / f"{pair_name}_leiden_modules.tsv"
    wto_edges_file = branch_b_dir / pair_name / f"{pair_name}_wTO_edges.tsv"

    if not edges_file.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_file}")
    if not modules_file.exists():
        raise FileNotFoundError(f"Missing Louvain modules file: {modules_file}")

    edges_df = pd.read_csv(edges_file)
    n_source = int(edges_df.shape[0])

    if int(args.viz_top_k) > 0:
        rank_col = "selected_value" if "selected_value" in edges_df.columns else "weight"
        edges_df = edges_df.sort_values(rank_col, ascending=False).head(int(args.viz_top_k)).reset_index(drop=True)

    g = build_graph(edges_df)
    # community assignments from script 31 — ensures figure is consistent with thesis tables
    node_to_module = read_leiden_modules(modules_file)

    # load wTO edges for the distribution plot (uses wTO_edges.tsv which has all wTO values)
    if wto_edges_file.exists():
        wto_df = pd.read_csv(wto_edges_file, sep="\t")
    else:
        wto_df = edges_df

    pair_out = out_root / pair_name / "branchB_permutation"
    pair_out.mkdir(parents=True, exist_ok=True)

    # combined figure: community layout + top-3 modules
    combined_png = pair_out / f"{pair_name}_branchB_permutation_network_combined.png"
    draw_combined_figure(
        g=g, pair_name=pair_name, node_to_module=node_to_module,
        out_png=combined_png, dpi=int(args.dpi), seed=int(args.seed),
        top_label_n=int(args.top_label_n),
    )

    # interactive HTML
    html_path: Path | None = None
    if bool(args.interactive_html):
        html_path = pair_out / f"{pair_name}_branchB_permutation_network_interactive.html"
        draw_html(g, html_path, title=f"{pair_name}: differential co-expression network", node_to_module=node_to_module)

    # degree distribution with Poisson null
    deg_png = pair_out / f"{pair_name}_branchB_permutation_degree_distribution.png"
    draw_degree_distribution(g, deg_png, pair_name, dpi=int(args.dpi))

    # wTO edge weight distribution
    wto_png = pair_out / f"{pair_name}_branchB_permutation_wto_distribution.png"
    draw_wto_distribution(wto_df, wto_png, pair_name, dpi=int(args.dpi))

    summary = {
        "pair": pair_name,
        "branch": "branchB_permutation",
        "n_edges_source": n_source,
        "n_nodes_visualized": g.number_of_nodes(),
        "n_edges_visualized": g.number_of_edges(),
        "combined_png": str(combined_png),
        "html": str(html_path) if html_path else "",
        "deg_png": str(deg_png),
        "wto_png": str(wto_png),
    }
    (pair_out / f"{pair_name}_branchB_permutation_viz_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[{pair_name}] nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    b_root = resolve_base(args.branch_b_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = sorted([p.name for p in b_root.iterdir() if p.is_dir() and "__vs__" in p.name])
    if args.pair:
        keep = set(args.pair)
        pairs = [p for p in pairs if p in keep]
    if not pairs:
        raise ValueError("No pair directories found in branch-b-dir")

    rows: list[dict[str, object]] = []
    for pair_name in pairs:
        print(f"\n--- {pair_name} ---")
        rows.append(run_pair(pair_name, b_root, out_root, args))

    pd.DataFrame(rows).sort_values("pair").to_csv(out_root / "network_viz_index.csv", index=False)
    print(f"\nDone. Outputs: {out_root}")


if __name__ == "__main__":
    main()
