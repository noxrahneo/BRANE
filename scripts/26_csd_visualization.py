#!/usr/bin/env python3
# flake8: noqa: E501
"""Union network zone visualization for CSD differential network outputs.

Figure layout per condition pair (large portrait format):
─────────────────────────────────────────────────────────────────────────────
│ Case M1 │ Case M2 │                                  │ Ctrl M1 │ Ctrl M2 │
│ Case M3 │ Case M4 │        FULL NETWORK              │ Ctrl M3 │ Ctrl M4 │
│ Case M5 │ Case M6 │  (community layout, module cols) │ Ctrl M5 │ Ctrl M6 │
─────────────────────────────────────────────────────────────────────────────
│          Shared M1          │ Shared M2 │          Shared M3              │
│          Shared M4          │ Shared M5 │          Shared M6              │
─────────────────────────────────────────────────────────────────────────────

Full network:  community layout (communities on a circle, spring within each).
               Nodes coloured by Leiden module; sized by weighted degree.
               Top-hub gene labels shown.
               Edges coloured by CSD type.
Module panels: spring layout; nodes coloured by zone (case/ctrl/shared).
               Gene labels on top hubs.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils.network_utils import resolve_base

#visual constants
EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}

_MODULE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d3d3d3", "#888888",
]

ZONE_PALETTE = {"case_only": "#f28e2b", "ctrl_only": "#76b7b2", "shared": "#4e79a7"}

MODULE_EDGE_FILTER_THRESHOLD = 600   # filter dense module panels to top-25% wTO
MAX_MODULE_NODES = 200               # cap spring layout per module panel
MIN_MODULE_DISPLAY_SIZE = 15         # full network: only show modules with ≥ this many nodes
MIN_ZONE_MODULE_SIZE = 5             # zone panels: minimum module size to include in layout
TOP_LABEL_FULL = 20                  # hub genes labelled in full network
TOP_LABEL_MODULE = 8                 # hub genes labelled in each module panel


#helpers
def module_color(mod_id: int) -> str:
    return _MODULE_PALETTE[(int(mod_id) - 1) % len(_MODULE_PALETTE)]


def leiden_on_graph(g: nx.Graph, seed: int = 7) -> dict[str, int]:
    if g.number_of_nodes() < 2 or g.number_of_edges() < 1:
        return {str(n): 1 for n in g.nodes()}
    nodes = list(g.nodes())
    idx   = {n: i for i, n in enumerate(nodes)}
    ig_g  = ig.Graph(
        n=len(nodes),
        edges=[(idx[u], idx[v]) for u, v in g.edges()],
        edge_attrs={"weight": [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]},
    )
    partition = leidenalg.find_partition(
        ig_g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed,
    )
    communities = sorted(
        [frozenset(nodes[i] for i in c) for c in partition], key=len, reverse=True,
    )
    result: dict[str, int] = {}
    for mod_id, comm in enumerate(communities, start=1):
        for gene in comm:
            result[str(gene)] = mod_id
    return result


def top_module_list(g: nx.Graph, mod_map: dict[str, int], n: int) -> list[tuple[int, list[str]]]:
    grps: dict[int, list[str]] = defaultdict(list)
    for gene, mid in mod_map.items():
        if gene in g.nodes():
            grps[mid].append(gene)
    return sorted(grps.items(), key=lambda kv: len(kv[1]), reverse=True)[:n]


def community_layout(g: nx.Graph, node_to_module: dict[str, int], seed: int = 7) -> dict[str, tuple[float, float]]:
    #spring layout on contracted meta-graph for module centroids; spring within each module
    buckets: dict[int, list[str]] = {}
    for node, mod in node_to_module.items():
        if node in g:
            buckets.setdefault(mod, []).append(node)
    mods = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_mods = len(mods)
    if n_mods == 0:
        return nx.spring_layout(g, seed=seed)
    if n_mods == 1:
        mod_id, nodes = mods[0]
        sub_pos = nx.spring_layout(g.subgraph(nodes), seed=seed, weight="weight", iterations=80)
        return dict(sub_pos)

    #build contracted meta-graph: one node per module, edges = total inter-module weight
    node_to_mod = {n: m for m, nodes in mods for n in nodes}
    meta = nx.Graph()
    for m, _ in mods:
        meta.add_node(m)
    for u, v, d in g.edges(data=True):
        mu = node_to_mod.get(u)
        mv = node_to_mod.get(v)
        if mu is not None and mv is not None and mu != mv:
            w = d.get("weight", 1.0)
            if meta.has_edge(mu, mv):
                meta[mu][mv]["weight"] += w
            else:
                meta.add_edge(mu, mv, weight=w)

    spread = max(100.0, n_mods * 18.0)
    k_meta = spread / math.sqrt(n_mods)
    meta_pos_raw = nx.spring_layout(meta, seed=seed, weight="weight", k=k_meta, iterations=150)

    #rescale meta positions to fill spread
    xs = [x for x, y in meta_pos_raw.values()]
    ys = [y for x, y in meta_pos_raw.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    scale = spread / max(x_range, y_range)
    centres: dict[int, tuple[float, float]] = {m: (x * scale, y * scale) for m, (x, y) in meta_pos_raw.items()}

    #module footprint: allow up to 65% of avg inter-centroid gap
    max_mod_scale = spread / (1.5 * math.sqrt(n_mods))

    pos: dict[str, tuple[float, float]] = {}
    for local_idx, (mod_id, nodes) in enumerate(mods):
        subg = g.subgraph(nodes)
        cx, cy = centres[mod_id]
        n = len(nodes)
        if n == 1:
            pos[nodes[0]] = (cx, cy)
        elif n == 2:
            pos[nodes[0]] = (cx - 0.5, cy)
            pos[nodes[1]] = (cx + 0.5, cy)
        else:
            k_param      = 18.0 / math.sqrt(n)
            module_scale = min(max_mod_scale, max(4.0, math.sqrt(n) * 2.0))
            sub_pos = nx.spring_layout(subg, seed=seed + local_idx, weight="weight", k=k_param, iterations=80)
            for node, (x, y) in sub_pos.items():
                pos[node] = (cx + x * module_scale, cy + y * module_scale)
    return pos


def filter_dense_edges(subg: nx.Graph) -> nx.Graph:
    if subg.number_of_edges() <= MODULE_EDGE_FILTER_THRESHOLD:
        return subg
    wto_vals = [d.get("wTO", float("nan")) for _, _, d in subg.edges(data=True)]
    wto_arr  = np.array(wto_vals, dtype=float)
    finite   = wto_arr[np.isfinite(wto_arr)]
    if finite.size == 0:
        weights   = np.array([d.get("weight", 0.0) for _, _, d in subg.edges(data=True)], dtype=float)
        threshold = float(np.percentile(weights, 75))
        keep = [(u, v) for (u, v, d), w in zip(subg.edges(data=True), weights) if w >= threshold]
    else:
        threshold = float(np.percentile(finite, 75))
        keep = [(u, v) for u, v, d in subg.edges(data=True) if float(d.get("wTO", float("nan"))) >= threshold]
    return subg.edge_subgraph(keep).copy()


def degree_scaled_sizes(g: nx.Graph, lo: float = 20.0, hi: float = 180.0) -> dict[str, float]:
    dw = dict(g.degree(weight="weight"))
    vals = np.array(list(dw.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        return {n: lo + (hi - lo) * (v - vmin) / (vmax - vmin) for n, v in dw.items()}
    return {n: (lo + hi) / 2 for n in dw}


def top_hub_genes(g: nx.Graph, n: int) -> list[str]:
    dw = dict(g.degree(weight="weight"))
    return sorted(dw, key=lambda x: dw[x], reverse=True)[:n]


#draw: full network panel
def draw_full_network(
    ax: plt.Axes,
    g_full: nx.Graph,
    node_to_module: dict[str, int],
    seed: int = 7,
    heading: str = "",
) -> None:
    if g_full.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    #keep only nodes belonging to modules large enough to be meaningful
    mod_sizes: dict[int, int] = {}
    for mod in node_to_module.values():
        mod_sizes[mod] = mod_sizes.get(mod, 0) + 1

    large_mod_ids = {mid for mid, sz in mod_sizes.items() if sz >= MIN_MODULE_DISPLAY_SIZE}
    display_nodes = {n for n, mid in node_to_module.items() if mid in large_mod_ids and n in g_full}

    g_display = g_full.subgraph(display_nodes).copy()
    #remove singletons and 2-node components after module filter
    small_comp_nodes = [n for comp in nx.connected_components(g_display) if len(comp) <= 2 for n in comp]
    if small_comp_nodes:
        g_display = g_display.copy()
        g_display.remove_nodes_from(small_comp_nodes)
    n_small_removed = len(small_comp_nodes)
    mod_display = {n: m for n, m in node_to_module.items() if n in g_display}

    n_dropped = g_full.number_of_nodes() - g_display.number_of_nodes()
    n_mods_shown = len(large_mod_ids)
    print(
        f"    full network display: {g_display.number_of_nodes()} nodes in {n_mods_shown} modules"
        f"  ({n_dropped} nodes hidden, {n_small_removed} singletons/pairs excluded)",
        flush=True,
    )

    print(f"    community layout…", flush=True)
    pos = community_layout(g_display, mod_display, seed=seed)

    #draw all edges within the display subgraph (modules are already filtered)
    for csd_type, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, d in g_display.edges(data=True) if d.get("link_type") == csd_type]
        if elist:
            nx.draw_networkx_edges(g_display, pos, edgelist=elist, ax=ax,
                                   edge_color=color, alpha=0.25, width=0.5)

    #nodes coloured by leiden module, sized by weighted degree
    sizes     = degree_scaled_sizes(g_display, lo=20.0, hi=220.0)
    node_list = list(g_display.nodes())
    colors    = [module_color(mod_display.get(n, 0)) for n in node_list]
    sz        = [sizes.get(n, 50.0) for n in node_list]
    nx.draw_networkx_nodes(g_display, pos, nodelist=node_list, ax=ax,
                           node_color=colors, node_size=sz, alpha=0.88,
                           linewidths=0.3, edgecolors="#ffffff")

    #hub gene labels for top hubs within the display graph
    hubs   = top_hub_genes(g_display, TOP_LABEL_FULL)
    labels = {n: n for n in hubs if n in pos}
    nx.draw_networkx_labels(g_display, pos, labels=labels, ax=ax,
                            font_size=5.5, font_color="#1d3557")

    stats_line = (
        f"Full union network  (modules ≥{MIN_MODULE_DISPLAY_SIZE} nodes shown)\n"
        f"total: {g_full.number_of_nodes():,} nodes · {g_full.number_of_edges():,} edges  |  "
        f"displayed: {g_display.number_of_nodes():,} nodes · {g_display.number_of_edges():,} edges · {n_mods_shown} modules"
    )
    if n_small_removed:
        stats_line += f"  |  {n_small_removed} singletons/pairs excluded"
    title_text = (heading + "\n" + stats_line) if heading else stats_line
    ax.set_title(title_text, fontsize=8.5, pad=5)
    #tighten axis limits to actual data extent
    all_x = [v[0] for v in pos.values()]
    all_y = [v[1] for v in pos.values()]
    x_pad = (max(all_x) - min(all_x)) * 0.04
    y_pad = (max(all_y) - min(all_y)) * 0.04
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
    ax.axis("off")


#draw: single module panel
def draw_module_panel(
    ax: plt.Axes,
    g_zone: nx.Graph,
    module_genes: list[str],
    zone: str,
    module_id: int,
    mod_map: dict[str, int],
    seed: int = 7,
) -> None:
    sg = g_zone.subgraph(module_genes).copy()
    if sg.number_of_nodes() == 0:
        ax.axis("off")
        return

    n_total = sg.number_of_nodes()
    if n_total > MAX_MODULE_NODES:
        top_nodes = sorted(sg.degree(weight="weight"), key=lambda kv: kv[1], reverse=True)[:MAX_MODULE_NODES]
        sg = sg.subgraph([n for n, _ in top_nodes]).copy()

    sg = filter_dense_edges(sg)

    pos   = nx.spring_layout(sg, seed=seed, weight="weight")
    sizes = degree_scaled_sizes(sg, lo=15.0, hi=120.0)
    node_list = list(sg.nodes())

    #nodes coloured by sub-module leiden within zone
    sub_mods = leiden_on_graph(sg, seed=seed)
    colors = [module_color(sub_mods.get(n, 1)) for n in node_list]
    sz     = [sizes.get(n, 30.0) for n in node_list]

    for csd_type, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, d in sg.edges(data=True) if d.get("link_type") == csd_type]
        if elist:
            nx.draw_networkx_edges(sg, pos, edgelist=elist, ax=ax,
                                   edge_color=color, alpha=0.45, width=0.7)

    nx.draw_networkx_nodes(sg, pos, nodelist=node_list, ax=ax,
                           node_color=colors, node_size=sz, alpha=0.88,
                           linewidths=0.4, edgecolors="#ffffff")

    hubs   = top_hub_genes(sg, TOP_LABEL_MODULE)
    labels = {n: n for n in hubs}
    nx.draw_networkx_labels(sg, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color="#1d3557")

    n_shown = sg.number_of_nodes()
    suffix  = f" /{n_total}" if n_shown < n_total else ""
    zone_short = {"case_only": "case", "ctrl_only": "ctrl", "shared": "shared"}.get(zone, zone)
    ax.set_title(f"M{module_id} [{zone_short}]  n={n_shown}{suffix}", fontsize=9, pad=3)
    ax.axis("off")


#standalone complete network: three spatial zones
def _zone_spring_layout(
    g: nx.Graph,
    cx: float,
    cy: float,
    seed: int,
    zone_radius: float,
) -> dict[str, tuple[float, float]]:
    #spring layout normalised and offset to (cx, cy); generous k so dense zones stay readable
    if g.number_of_nodes() == 0:
        return {}
    n = g.number_of_nodes()
    #larger k than default → more spread, inversely scaled with density
    k = max(0.8, 3.5 / math.sqrt(n))
    if g.number_of_edges() == 0:
        raw = nx.spring_layout(g, seed=seed, k=k, iterations=1)
    else:
        raw = nx.spring_layout(g, seed=seed, weight="weight", k=k, iterations=100)

    #normalise then scale to zone_radius
    xs = [x for x, y in raw.values()]
    ys = [y for x, y in raw.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    scale = zone_radius / max(x_range, y_range) * 2.0
    x_mid = (max(xs) + min(xs)) / 2
    y_mid = (max(ys) + min(ys)) / 2

    return {
        node: (cx + (x - x_mid) * scale, cy + (y - y_mid) * scale)
        for node, (x, y) in raw.items()
    }


def _thin_intra_edges(
    g_full: nx.Graph,
    zone_genes: set[str],
    top_quantile: float = 0.70,
) -> list[tuple[str, str, dict]]:
    #return intra-zone edges filtered to top (1-top_quantile) by weight to avoid overplotting
    edges = [(u, v, d) for u, v, d in g_full.edges(data=True)
             if u in zone_genes and v in zone_genes]
    if not edges:
        return []
    weights = np.array([d.get("weight", 0.0) for _, _, d in edges], dtype=float)
    threshold = float(np.quantile(weights, top_quantile))
    return [(u, v, d) for (u, v, d), w in zip(edges, weights) if w >= threshold]


def _draw_zone_panel(
    ax: plt.Axes,
    g_zone: nx.Graph,
    zone: str,
    label: str,
    seed: int,
    top_edge_quantile: float = 0.60,
) -> dict[str, tuple[float, float]]:
    #draw one zone panel with community layout; returns pos dict for all zone nodes
    color = ZONE_PALETTE[zone]
    ax.set_facecolor("#f7f7f7")

    if g_zone.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return {}

    #community detection on zone subgraph
    mod_map = leiden_on_graph(g_zone, seed=seed)

    #filter to modules with ≥ MIN_ZONE_MODULE_SIZE nodes to avoid ring-of-dots artifact
    mod_sizes: dict[int, int] = {}
    for n, m in mod_map.items():
        mod_sizes[m] = mod_sizes.get(m, 0) + 1
    keep_mods = {m for m, sz in mod_sizes.items() if sz >= MIN_ZONE_MODULE_SIZE}
    display_nodes = {n for n, m in mod_map.items() if m in keep_mods}
    n_hidden = g_zone.number_of_nodes() - len(display_nodes)

    g_display = g_zone.subgraph(display_nodes) if display_nodes else g_zone
    mod_display = {n: m for n, m in mod_map.items() if n in display_nodes}

    if g_display.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No modules", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return {}

    pos = community_layout(g_display, mod_display, seed=seed)

    #only draw top edges to avoid solid-mass overplotting
    if g_display.number_of_edges() > 0:
        all_e = list(g_display.edges(data=True))
        weights = np.array([d.get("weight", 0.0) for _, _, d in all_e], dtype=float)
        threshold = float(np.quantile(weights, top_edge_quantile))
        for csd_type, ec in EDGE_COLORS.items():
            elist = [(u, v) for (u, v, d), w in zip(all_e, weights)
                     if d.get("link_type") == csd_type and w >= threshold]
            if elist:
                nx.draw_networkx_edges(g_display, pos, edgelist=elist, ax=ax,
                                       edge_color=ec, alpha=0.40, width=0.7)

    sizes = degree_scaled_sizes(g_display, lo=15.0, hi=200.0)
    node_list = list(g_display.nodes())
    node_colors = [module_color(mod_display.get(n, 1)) for n in node_list]
    sz = [sizes.get(n, 25.0) for n in node_list]

    nx.draw_networkx_nodes(g_display, pos, nodelist=node_list, ax=ax,
                           node_color=node_colors, node_size=sz,
                           alpha=0.90, linewidths=0.3, edgecolors="#ffffff")

    hubs = top_hub_genes(g_display, TOP_LABEL_FULL)
    nx.draw_networkx_labels(g_display, pos, labels={n: n for n in hubs}, ax=ax,
                            font_size=9, font_color="#1d3557", font_weight="bold")

    n_mods = len(keep_mods)
    hidden_note = f" · {n_hidden} small modules hidden" if n_hidden > 0 else ""
    ax.set_title(
        f"{label}\n{len(display_nodes):,} nodes · {n_mods} modules{hidden_note}\n"
        f"(top {int((1-top_edge_quantile)*100)}% edges shown)",
        fontsize=12, color=color, fontweight="bold", pad=6,
    )
    ax.axis("off")
    return pos


def draw_standalone_network(
    g_full: nx.Graph,
    node_to_module: dict[str, int],
    pres_map: dict[str, str],
    pair_name: str,
    case: str,
    ctrl: str,
    out_path: Path,
    dpi: int = 150,
    seed: int = 7,
) -> None:
    #three-panel standalone: case | shared | ctrl with community layout and cross-zone edge lines
    nodes_in_net = set(g_full.nodes())
    case_genes   = {g for g in nodes_in_net if pres_map.get(g) == "case_only"}
    ctrl_genes   = {g for g in nodes_in_net if pres_map.get(g) == "ctrl_only"}
    shared_genes = {g for g in nodes_in_net if pres_map.get(g) == "shared"}

    g_case   = g_full.subgraph(case_genes).copy()
    g_ctrl   = g_full.subgraph(ctrl_genes).copy()
    g_shared = g_full.subgraph(shared_genes).copy()

    print(
        f"    3-panel layouts "
        f"(case={len(case_genes)} | shared={len(shared_genes)} | ctrl={len(ctrl_genes)})…",
        flush=True,
    )

    #1 row x 3 cols, wide landscape
    fig, axes = plt.subplots(1, 3, figsize=(48, 20))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        f"Complete Union Network — Zone Panels\n"
        f"{pair_name.replace('__vs__', ' vs ')}  |  "
        f"total nodes={g_full.number_of_nodes():,}  edges={g_full.number_of_edges():,}",
        fontsize=16, y=1.00,
    )

    #draw each zone panel; shared zone slightly stricter edge quantile to avoid overplotting
    pos_case   = _draw_zone_panel(axes[0], g_case,   "case_only",
                                  f"Case-specific — {case}",   seed,     top_edge_quantile=0.40)
    pos_shared = _draw_zone_panel(axes[1], g_shared, "shared",
                                  "Shared genes",               seed + 1, top_edge_quantile=0.55)
    pos_ctrl   = _draw_zone_panel(axes[2], g_ctrl,   "ctrl_only",
                                  f"Ctrl-specific — {ctrl}",   seed + 2, top_edge_quantile=0.40)

    #cross-zone edges as inter-panel connection lines via ConnectionPatch
    from matplotlib.patches import ConnectionPatch
    cross_edges = [
        (u, v, d) for u, v, d in g_full.edges(data=True)
        if not (u in case_genes and v in case_genes)
        and not (u in ctrl_genes and v in ctrl_genes)
        and not (u in shared_genes and v in shared_genes)
    ]
    #only draw top 300 cross-zone edges to avoid overplotting
    cross_edges_sorted = sorted(cross_edges, key=lambda e: e[2].get("weight", 0), reverse=True)[:300]

    zone_ax   = {"case_only": axes[0], "ctrl_only": axes[2], "shared": axes[1]}
    zone_pos  = {"case_only": pos_case, "ctrl_only": pos_ctrl, "shared": pos_shared}
    zone_gene = {"case_only": case_genes, "ctrl_only": ctrl_genes, "shared": shared_genes}

    for u, v, d in cross_edges_sorted:
        link_type = d.get("link_type", "S")
        color     = EDGE_COLORS.get(link_type, "#888888")
        u_zone    = pres_map.get(u, "shared")
        v_zone    = pres_map.get(v, "shared")
        if u_zone == v_zone:
            continue
        pu = zone_pos.get(u_zone, {}).get(u)
        pv = zone_pos.get(v_zone, {}).get(v)
        if pu is None or pv is None:
            continue
        cp = ConnectionPatch(
            xyA=pu, coordsA="data", axesA=zone_ax[u_zone],
            xyB=pv, coordsB="data", axesB=zone_ax[v_zone],
            color=color, alpha=0.15, linewidth=0.4, zorder=0,
        )
        fig.add_artist(cp)

    #legend
    edge_patches = [mpatches.Patch(color=c, label=f"{t} edge") for t, c in EDGE_COLORS.items()]
    zone_patches = [mpatches.Patch(color=ZONE_PALETTE[z], label=z.replace("_", " "))
                    for z in ["case_only", "shared", "ctrl_only"]]
    cross_note  = [mpatches.Patch(color="#888888", alpha=0.4,
                                  label=f"cross-zone edges (top 300 shown)")]
    fig.legend(handles=edge_patches + zone_patches + cross_note,
               loc="lower center", fontsize=9, ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), title="Edge type / zone", title_fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  standalone saved → {out_path}")


#assemble figure
def make_figure(
    pair_name: str,
    case: str,
    ctrl: str,
    g_full: nx.Graph,
    g_case: nx.Graph,
    g_ctrl: nx.Graph,
    g_shared: nx.Graph,
    pres_map: dict[str, str],
    node_to_module_full: dict[str, int],
    mods_case: list[tuple[int, list[str]]],
    mods_ctrl: list[tuple[int, list[str]]],
    mods_shared: list[tuple[int, list[str]]],
    out_path: Path,
    dpi: int = 150,
    seed: int = 7,
) -> None:
    #large portrait; side columns wide enough for readable module panels
    fig = plt.figure(figsize=(36, 44))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        f"Differential Co-expression Network — Union (Branch B Permutation)\n"
        f"{pair_name.replace('__vs__', ' vs ')}",
        fontsize=15, y=0.999, va="top",
    )

    #outer grid: top (network + side modules) / bottom (shared modules)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.0, 1.6],
        hspace=0.20,
        top=0.975, bottom=0.03, left=0.02, right=0.98,
    )

    #top: case 2-col | full network | ctrl 2-col; side columns wider for readability
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0],
        width_ratios=[1.4, 2.2, 1.4],
        wspace=0.06,
    )
    case_gs  = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=top_gs[0], hspace=0.42, wspace=0.32)
    full_ax  = fig.add_subplot(top_gs[1])
    ctrl_gs  = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=top_gs[2], hspace=0.42, wspace=0.32)

    #bottom: shared 2x3 grid
    bot_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], hspace=0.40, wspace=0.28)

    case_axes   = [fig.add_subplot(case_gs[r, c]) for r in range(3) for c in range(2)]
    ctrl_axes   = [fig.add_subplot(ctrl_gs[r, c]) for r in range(3) for c in range(2)]
    shared_axes = [fig.add_subplot(bot_gs[r, c])  for r in range(2) for c in range(3)]

    #zone labels
    fig.text(0.095, 0.978, f"Case-specific\n{case}",
             ha="center", fontsize=10, color=ZONE_PALETTE["case_only"], fontweight="bold")
    fig.text(0.905, 0.978, f"Ctrl-specific\n{ctrl}",
             ha="center", fontsize=10, color=ZONE_PALETTE["ctrl_only"], fontweight="bold")
    fig.text(0.500, 0.268, "Shared genes",
             ha="center", fontsize=10, color=ZONE_PALETTE["shared"], fontweight="bold")

    #draw full network and module panels
    draw_full_network(full_ax, g_full, node_to_module_full, seed=seed)

    for idx, ax in enumerate(case_axes):
        if idx < len(mods_case):
            mid, genes = mods_case[idx]
            draw_module_panel(ax, g_case, genes, "case_only", mid, {}, seed)
        else:
            ax.axis("off")

    for idx, ax in enumerate(ctrl_axes):
        if idx < len(mods_ctrl):
            mid, genes = mods_ctrl[idx]
            draw_module_panel(ax, g_ctrl, genes, "ctrl_only", mid, {}, seed)
        else:
            ax.axis("off")

    for idx, ax in enumerate(shared_axes):
        if idx < len(mods_shared):
            mid, genes = mods_shared[idx]
            draw_module_panel(ax, g_shared, genes, "shared", mid, {}, seed)
        else:
            ax.axis("off")

    #module colour legend for full network
    n_mods_full = len(set(node_to_module_full.values()))
    mod_patches = [
        mpatches.Patch(color=module_color(i), label=f"Module {i}")
        for i in range(1, min(n_mods_full + 1, 13))
    ]
    if n_mods_full > 12:
        mod_patches.append(mpatches.Patch(color="#888888", label=f"… +{n_mods_full - 12} more"))

    edge_patches = [mpatches.Patch(color=c, label=f"{t} edge") for t, c in EDGE_COLORS.items()]

    fig.legend(handles=edge_patches,
               loc="lower left", fontsize=7, ncol=1,
               bbox_to_anchor=(0.01, 0.005), frameon=False, title="Edge type", title_fontsize=7)
    fig.legend(handles=mod_patches,
               loc="lower right", fontsize=6.5, ncol=2,
               bbox_to_anchor=(0.99, 0.005), frameon=False, title="Leiden modules (full network)", title_fontsize=7)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {out_path}")


#per-pair driver helpers
def _save_full_network_standalone(
    g_full: nx.Graph,
    node_to_module: dict[str, int],
    pair_name: str,
    case: str,
    ctrl: str,
    out_path: Path,
    dpi: int,
    seed: int,
) -> None:
    #save full network community layout as a standalone portrait figure
    fig, ax = plt.subplots(figsize=(18, 24))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.18)
    fig.patch.set_facecolor("#fafafa")
    draw_full_network(ax, g_full, node_to_module, seed=seed,
                      heading=f"Differential Co-expression Network — Union  |  {pair_name.replace('__vs__', ' vs ')}")
    #edge-type legend bottom left; module legend bottom right
    edge_patches = [mpatches.Patch(color=c, label=f"{t} — {'Conserved' if t == 'C' else 'Specific' if t == 'S' else 'Differentiated'}")
                    for t, c in EDGE_COLORS.items()]
    fig.legend(handles=edge_patches, loc="lower left", fontsize=9, ncol=1,
               bbox_to_anchor=(0.01, 0.01), frameon=True, framealpha=0.85,
               title="Edge type", title_fontsize=9)
    n_mods = len(set(node_to_module.values()))
    mod_patches = [mpatches.Patch(color=module_color(i), label=f"Module {i}") for i in range(1, min(n_mods + 1, 13))]
    if n_mods > 12:
        mod_patches.append(mpatches.Patch(color="#888888", label=f"… +{n_mods - 12} more"))
    mod_patches.append(mpatches.Patch(color="none", label="Colors independent\nacross figures"))
    fig.legend(handles=mod_patches, loc="lower right", fontsize=8, ncol=3,
               bbox_to_anchor=(0.99, 0.01), frameon=True, framealpha=0.85,
               title="Leiden modules (within-figure only)", title_fontsize=8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def _save_zone_module_figure(
    g_zone: nx.Graph,
    mods: list[tuple[int, list[str]]],
    zone: str,
    zone_label: str,
    pair_name: str,
    out_path: Path,
    dpi: int,
    seed: int,
) -> None:
    #3x2 grid of zone module panels
    zone_color = ZONE_PALETTE.get(zone, "#333333")

    fig = plt.figure(figsize=(24, 17))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.12,
                            top=0.92, bottom=0.03, left=0.03, right=0.97)

    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        if i >= len(mods):
            ax.axis("off")
            continue

        mod_id, mod_nodes = mods[i]
        mod_nodes_in_g = [n for n in mod_nodes if n in g_zone]
        if not mod_nodes_in_g:
            ax.axis("off")
            continue

        sg = g_zone.subgraph(mod_nodes_in_g).copy()
        n_total = sg.number_of_nodes()
        if n_total > MAX_MODULE_NODES:
            top_by_deg = sorted(sg.degree(weight="weight"), key=lambda kv: kv[1], reverse=True)[:MAX_MODULE_NODES]
            sg = sg.subgraph([n for n, _ in top_by_deg]).copy()
        sg = filter_dense_edges(sg)
        small_comp_nodes = [n for comp in nx.connected_components(sg) if len(comp) <= 2 for n in comp]
        if small_comp_nodes:
            sg.remove_nodes_from(small_comp_nodes)
        n_excluded = len(small_comp_nodes)

        if sg.number_of_nodes() == 0:
            ax.set_title(f"Module {i + 1}  [{zone_label}]\n(no connected genes after filtering)", fontsize=9, pad=5, color=zone_color)
            ax.axis("off")
            continue

        k_val  = 4.0 / max(1, math.sqrt(len(mod_nodes_in_g)))
        sub_pos = nx.spring_layout(sg, seed=seed + i, weight="weight", k=k_val, iterations=80)
        sub_mods = leiden_on_graph(sg, seed=seed)

        for csd_type, ec in EDGE_COLORS.items():
            elist = [(u, v) for u, v, d in sg.edges(data=True) if d.get("link_type") == csd_type]
            if elist:
                nx.draw_networkx_edges(sg, sub_pos, edgelist=elist, ax=ax,
                                       edge_color=ec, alpha=0.50, width=0.8)

        colors = [module_color(sub_mods.get(n, 1)) for n in sg.nodes()]
        sizes  = degree_scaled_sizes(sg, lo=15.0, hi=130.0)
        nx.draw_networkx_nodes(sg, sub_pos, nodelist=list(sg.nodes()), ax=ax,
                               node_color=colors, node_size=[sizes.get(n, 30) for n in sg.nodes()],
                               alpha=0.90, linewidths=0.4, edgecolors="#ffffff")

        hubs = top_hub_genes(sg, TOP_LABEL_MODULE)
        if hubs:
            nx.draw_networkx_labels(sg, sub_pos, labels={n: n for n in hubs}, ax=ax,
                                    font_size=7, font_color="#1d3557", font_weight="bold")

        #3-line title: size note, edges, top hubs
        n_shown  = sg.number_of_nodes()
        n_edges  = sg.number_of_edges()
        top3     = hubs[:3]
        size_note = f"{n_shown} nodes" + (f" / {n_total} total" if n_shown < n_total else "")
        if n_excluded:
            size_note += f"  ({n_excluded} singletons/pairs excluded)"
        hub_line  = f"Top hubs: {', '.join(top3)}" if top3 else "—"
        edge_line = f"{n_edges} intra-module edges"

        ax.set_title(
            f"Module {i + 1}  [{zone_label}]\n{size_note}  ·  {edge_line}\n{hub_line}",
            fontsize=10, pad=6, color=zone_color, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"Differential Co-expression — {zone_label} Modules\n"
        f"{pair_name.replace('__vs__', ' vs ')}",
        fontsize=13, color=zone_color, fontweight="bold",
    )
    edge_patches = [mpatches.Patch(color=c, label=f"{t} — {'Conserved' if t == 'C' else 'Specific' if t == 'S' else 'Differentiated'}")
                    for t, c in EDGE_COLORS.items()]
    fig.legend(handles=edge_patches, loc="lower left", fontsize=9, ncol=3,
               bbox_to_anchor=(0.01, 0.00), frameon=True, framealpha=0.85,
               title="Edge type", title_fontsize=9)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)


def run_pair(
    pair_name: str,
    network_root: Path,
    upstream_root: Path,
    out_root: Path,
    dpi: int,
    seed: int,
) -> None:
    case, ctrl  = pair_name.split("__vs__", 1)
    edges_file  = network_root / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    pres_file   = upstream_root / pair_name / f"{pair_name}_gene_presence.csv"

    if not edges_file.exists():
        print(f"[warn] {pair_name}: missing edges file")
        return
    if not pres_file.exists():
        print(f"[warn] {pair_name}: missing gene_presence file")
        return

    print(f"[{pair_name}]")
    edges_df = pd.read_csv(edges_file)
    pres_df  = pd.read_csv(pres_file)
    pres_map = dict(zip(pres_df["gene"].astype(str), pres_df["presence"]))

    if edges_df.empty:
        print(f"  [warn] no edges, skipping")
        return

    #build full graph from edge table
    g_full = nx.Graph()
    for row in edges_df.itertuples(index=False):
        g_full.add_edge(
            str(row.gene_a), str(row.gene_b),
            weight=float(row.weight),
            link_type=str(row.link_type),
            C=float(row.C), S=float(row.S), D=float(row.D),
            wTO=float(row.wTO) if hasattr(row, "wTO") and not math.isnan(float(row.wTO)) else float("nan"),
        )

    #split into case/ctrl/shared zone subgraphs
    nodes_in_net = set(g_full.nodes())
    case_genes   = {g for g in nodes_in_net if pres_map.get(g) == "case_only"}
    ctrl_genes   = {g for g in nodes_in_net if pres_map.get(g) == "ctrl_only"}
    shared_genes = {g for g in nodes_in_net if pres_map.get(g) == "shared"}

    g_case   = g_full.subgraph(case_genes).copy()
    g_ctrl   = g_full.subgraph(ctrl_genes).copy()
    g_shared = g_full.subgraph(shared_genes).copy()

    #run leiden on full graph and each zone subgraph
    print(f"  Leiden: full graph…", flush=True)
    node_to_module_full = leiden_on_graph(g_full, seed=seed)
    print(f"  Leiden: zone subgraphs…", flush=True)
    mods_case   = top_module_list(g_case,   leiden_on_graph(g_case,   seed), 6)
    mods_ctrl   = top_module_list(g_ctrl,   leiden_on_graph(g_ctrl,   seed), 6)
    mods_shared = top_module_list(g_shared, leiden_on_graph(g_shared, seed), 6)

    n_full_mods = len(set(node_to_module_full.values()))
    print(
        f"  nodes: case={len(case_genes)} ctrl={len(ctrl_genes)} shared={len(shared_genes)}"
        f"  |  full_mods={n_full_mods}"
        f"  case_mods={len(mods_case)} ctrl_mods={len(mods_ctrl)} shared_mods={len(mods_shared)}"
    )

    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    #combined zone+modules figure
    make_figure(
        pair_name=pair_name, case=case, ctrl=ctrl,
        g_full=g_full, g_case=g_case, g_ctrl=g_ctrl, g_shared=g_shared,
        pres_map=pres_map,
        node_to_module_full=node_to_module_full,
        mods_case=mods_case, mods_ctrl=mods_ctrl, mods_shared=mods_shared,
        out_path=pair_out / f"{pair_name}_zone_modules.png", dpi=dpi, seed=seed,
    )

    #standalone 3-panel: case | shared | ctrl
    draw_standalone_network(
        g_full=g_full, node_to_module=node_to_module_full,
        pres_map=pres_map, pair_name=pair_name, case=case, ctrl=ctrl,
        out_path=pair_out / f"{pair_name}_complete_network.png", dpi=dpi, seed=seed,
    )

    #standalone full network (center only, no module panels)
    print("  full network standalone…", flush=True)
    _save_full_network_standalone(
        g_full=g_full, node_to_module=node_to_module_full,
        pair_name=pair_name, case=case, ctrl=ctrl,
        out_path=pair_out / f"{pair_name}_full_network.png", dpi=dpi, seed=seed,
    )

    #per-zone module figures (3x2 grid)
    print("  zone module figures…", flush=True)
    for zone, g_zone, mods, label in [
        ("case_only",  g_case,   mods_case,   f"Case — {case}"),
        ("shared",     g_shared, mods_shared, "Shared genes"),
        ("ctrl_only",  g_ctrl,   mods_ctrl,   f"Ctrl — {ctrl}"),
    ]:
        slug = zone.replace("_only", "")
        _save_zone_module_figure(
            g_zone=g_zone, mods=mods, zone=zone, zone_label=label,
            pair_name=pair_name,
            out_path=pair_out / f"{pair_name}_modules_{slug}.png",
            dpi=dpi, seed=seed,
        )


#cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Union network zone visualization")
    p.add_argument("--network-dir",  default="results/14_csd_networks")
    p.add_argument("--upstream-dir", default="results/12_csd_scoring")
    p.add_argument("--output-dir",   default="results/16_network_viz")
    p.add_argument("--pair", action="append", default=[])
    p.add_argument("--dpi",  type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> int:
    args     = parse_args()

    #resolve paths and create output dir
    net_root = resolve_base(args.network_dir)
    up_root  = resolve_base(args.upstream_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    #discover pair dirs and filter to requested subset if specified
    pair_dirs = sorted([p for p in net_root.iterdir() if p.is_dir() and "__vs__" in p.name])
    if args.pair:
        keep      = set(args.pair)
        pair_dirs = [p for p in pair_dirs if p.name in keep]
    if not pair_dirs:
        raise ValueError("No pair directories found")

    #visualize each pair
    for pair_dir in pair_dirs:
        run_pair(pair_name=pair_dir.name, network_root=net_root, upstream_root=up_root,
                 out_root=out_root, dpi=int(args.dpi), seed=int(args.seed))
    print(f"\nDone. Zone visualizations → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
