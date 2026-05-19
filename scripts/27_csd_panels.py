#!/usr/bin/env python3
# flake8: noqa: E501
"""CSD panel figures per condition pair.

Five publication-ready figures per condition pair:
  {pair}_shared_network.png   — shared genes, all C/S/D edges, all modules coloured
  {pair}_shared_C.png         — shared genes, C edges only (conserved co-expression)
  {pair}_shared_D.png         — shared genes, D edges only (reprogrammed co-expression)
  {pair}_specific_S_case.png  — all genes, tumour-gained S edges (co-expression gained in tumour)
  {pair}_specific_S_ctrl.png  — all genes, normal-retained S edges (co-expression lost in tumour)

C = Conserved   S = Specific   D = Differentiated

Input directories:
  --network-dir  results/14_csd_networks    (31a: edge CSVs with rho_case, rho_control)
  --shared-dir   results/12_csd_scoring  (28a: gene_presence.csv per pair)
  --output-dir   results/16_network_viz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils.network_utils import resolve_base

#visual constants
EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}
EDGE_LABELS = {"C": "C — Conserved", "S": "S — Specific", "D": "D — Differentiated"}

_BG_COLOR    = "#f9f9f9"
_SINGLETON_COLOR = "#d0d0d0"   # nodes with no module partners
_SINGLETON_SIZE  = 12.0

TOP_LABEL_GENES = 15           # hub gene labels per panel


def _make_palette(n: int) -> list[str]:
    #return n visually distinct hex colours, cycling tab20 for large n
    base = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
    ]
    if n <= len(base):
        return base[:n]
    cmap = plt.cm.get_cmap("hsv", n + 1)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


#graph helpers
def leiden_on_graph(g: nx.Graph, seed: int = 7) -> dict[str, int]:
    if g.number_of_nodes() < 2 or g.number_of_edges() < 1:
        return {str(n): 0 for n in g.nodes()}   # module 0 = singleton/no-module
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


def community_layout(g: nx.Graph, node_to_module: dict[str, int], seed: int = 7) -> dict[str, tuple[float, float]]:
    #spring layout on inter-module meta-graph for centroids; spring within each module
    buckets: dict[int, list[str]] = {}
    for node, mod in node_to_module.items():
        if node in g:
            buckets.setdefault(mod, []).append(node)

    #separate singletons (mod=0 or size-1) from proper modules; place singletons peripherally
    singleton_nodes: list[str] = []
    proper: list[tuple[int, list[str]]] = []
    for mod, nodes in buckets.items():
        if len(nodes) <= 1 or mod == 0:
            singleton_nodes.extend(nodes)
        else:
            proper.append((mod, nodes))
    proper = sorted(proper, key=lambda kv: len(kv[1]), reverse=True)

    if not proper:
        return nx.spring_layout(g, seed=seed, k=2.0 / math.sqrt(max(g.number_of_nodes(), 1)))

    if len(proper) == 1:
        mod_id, nodes = proper[0]
        pos = dict(nx.spring_layout(g.subgraph(nodes), seed=seed, weight="weight", iterations=80))
    else:
        #contracted meta-graph
        node_to_mod = {n: m for m, nodes in proper for n in nodes}
        meta = nx.Graph()
        for m, _ in proper:
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

        n_mods = len(proper)
        spread  = max(12.0, n_mods * 3.0)

        #if most modules have no inter-module edges, spring degenerates to a ring;
        #use random-disk init instead to break the repulsion-only symmetry
        meta_edges_nodes = set(u for u, v in meta.edges()) | set(v for u, v in meta.edges())
        frac_connected   = len(meta_edges_nodes) / n_mods if n_mods > 0 else 1.0
        k_meta = spread / math.sqrt(n_mods)
        if frac_connected < 0.35:
            rng = np.random.default_rng(seed)
            init_pos = {}
            for mod_id, _ in proper:
                angle  = rng.uniform(0, 2 * math.pi)
                radius = spread * 0.45 * math.sqrt(rng.uniform(0.05, 1.0))
                init_pos[mod_id] = np.array([radius * math.cos(angle), radius * math.sin(angle)])
            #40 iterations from random init: connected modules attract, disconnected don't ring
            meta_raw = nx.spring_layout(meta, pos=init_pos, seed=seed,
                                         weight="weight", k=k_meta, iterations=40)
        else:
            meta_raw = nx.spring_layout(meta, seed=seed, weight="weight", k=k_meta, iterations=150)
        xs = [x for x, y in meta_raw.values()]
        ys = [y for x, y in meta_raw.values()]
        x_range = max(xs) - min(xs) or 1.0
        y_range = max(ys) - min(ys) or 1.0
        scale    = spread / max(x_range, y_range)
        centres  = {m: (x * scale, y * scale) for m, (x, y) in meta_raw.items()}

        pos: dict[str, tuple[float, float]] = {}
        for local_idx, (mod_id, nodes) in enumerate(proper):
            subg = g.subgraph(nodes)
            cx, cy = centres[mod_id]
            n = len(nodes)
            if n == 1:
                pos[nodes[0]] = (cx, cy)
            elif n == 2:
                pos[nodes[0]] = (cx - 0.5, cy)
                pos[nodes[1]] = (cx + 0.5, cy)
            else:
                k_param      = 4.0 / math.sqrt(n)
                module_scale = max(4.0, math.sqrt(n) * 1.0)
                sub_pos = nx.spring_layout(subg, seed=seed + local_idx, weight="weight",
                                           k=k_param, iterations=80)
                for node, (x, y) in sub_pos.items():
                    pos[node] = (cx + x * module_scale, cy + y * module_scale)

    #place singletons around a slightly larger bounding circle
    if singleton_nodes and pos:
        xs_all = [x for x, y in pos.values()]
        ys_all = [y for x, y in pos.values()]
        cx_all = (max(xs_all) + min(xs_all)) / 2
        cy_all = (max(ys_all) + min(ys_all)) / 2
        radius = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)) / 2 * 1.25
        rng = np.random.default_rng(seed)
        for i, node in enumerate(singleton_nodes):
            angle = 2 * math.pi * i / max(len(singleton_nodes), 1)
            r = radius + rng.uniform(0.5, 2.0)
            pos[node] = (cx_all + r * math.cos(angle), cy_all + r * math.sin(angle))
    elif singleton_nodes:
        rng = np.random.default_rng(seed)
        for node in singleton_nodes:
            pos[node] = (rng.uniform(-5, 5), rng.uniform(-5, 5))

    return pos


def degree_scaled_sizes(g: nx.Graph, lo: float = 18.0, hi: float = 250.0) -> dict[str, float]:
    dw = dict(g.degree(weight="weight"))
    vals = np.array(list(dw.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        return {n: lo + (hi - lo) * (v - vmin) / (vmax - vmin) for n, v in dw.items()}
    return {n: (lo + hi) / 2 for n in dw}


def top_hub_genes(g: nx.Graph, n: int) -> list[str]:
    dw = dict(g.degree(weight="weight"))
    return sorted(dw, key=lambda x: dw[x], reverse=True)[:n]


#loading
def load_edges(edges_file: Path) -> pd.DataFrame:
    df = pd.read_csv(edges_file)
    df["rho_case"]    = pd.to_numeric(df["rho_case"],    errors="coerce").fillna(0.0)
    df["rho_control"] = pd.to_numeric(df["rho_control"], errors="coerce").fillna(0.0)
    return df


def build_graph(edges_df: pd.DataFrame, link_types: set[str],
                node_filter: set[str] | None = None) -> nx.Graph:
    g = nx.Graph()
    for _, row in edges_df.iterrows():
        if row["link_type"] not in link_types:
            continue
        u, v = str(row["gene_a"]), str(row["gene_b"])
        if node_filter is not None and (u not in node_filter or v not in node_filter):
            continue
        g.add_edge(u, v,
                   weight=float(row.get("weight", 1.0)),
                   link_type=str(row["link_type"]),
                   wTO=float(row.get("wTO", 0.0)),
                   rho_case=float(row["rho_case"]),
                   rho_control=float(row["rho_control"]))
    return g


def build_directional_s_graph(edges_df: pd.DataFrame, case_dominant: bool) -> nx.Graph:
    g = nx.Graph()
    for _, row in edges_df[edges_df["link_type"] == "S"].iterrows():
        rc = abs(float(row["rho_case"]))
        rk = abs(float(row["rho_control"]))
        if (rc >= rk) != case_dominant:
            continue
        u, v = str(row["gene_a"]), str(row["gene_b"])
        g.add_edge(u, v,
                   weight=float(row.get("weight", 1.0)),
                   link_type="S",
                   wTO=float(row.get("wTO", 0.0)),
                   rho_case=float(row["rho_case"]),
                   rho_control=float(row["rho_control"]))
    return g


#drawing
def component_layout(g: nx.Graph, node_to_module: dict[str, int], seed: int = 7,
                     min_blob_size: int = 5) -> dict[str, tuple[float, float]]:
    #layout for fragmented graphs: large components as blobs, small scattered peripherally
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    large = [(i, list(c)) for i, c in enumerate(components) if len(c) >= min_blob_size]
    small = [n for c in components if len(c) < min_blob_size for n in c]

    rng = np.random.default_rng(seed)

    if not large:
        #all singletons — just spread them
        k = max(1.0, 4.0 / math.sqrt(max(g.number_of_nodes(), 1)))
        return nx.spring_layout(g, seed=seed, weight="weight", k=k, iterations=80)

    #build meta-graph of large components; use size-weighted positions to drive spread
    n_large = len(large)
    spread  = max(12.0, math.sqrt(n_large) * 6.0)

    #initialise meta positions: largest component at centre, rest by size rank
    meta = nx.Graph()
    meta.add_nodes_from(range(n_large))
    init_meta: dict[int, tuple[float, float]] = {}
    ncols = max(1, math.ceil(math.sqrt(n_large)))
    for rank, (comp_idx, comp_nodes) in enumerate(large):
        col = rank % ncols
        row = rank // ncols
        init_meta[rank] = (
            (col - ncols / 2) * spread / ncols,
            -(row * spread / math.ceil(n_large / ncols)),
        )
    meta_pos = nx.spring_layout(meta, seed=seed, pos=init_meta, fixed=None,
                                k=spread / math.sqrt(n_large), iterations=60)

    #scale meta positions to fill spread
    xs = [x for x, y in meta_pos.values()]
    ys = [y for x, y in meta_pos.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    scale   = spread / max(x_range, y_range)
    centres = {rank: (x * scale, y * scale) for rank, (x, y) in meta_pos.items()}

    pos: dict[str, tuple[float, float]] = {}
    for rank, (comp_idx, comp_nodes) in enumerate(large):
        cx, cy   = centres[rank]
        subg     = g.subgraph(comp_nodes)
        n        = len(comp_nodes)
        mod_scale = max(2.5, math.sqrt(n) * 0.8)
        k_param   = 2.5 / math.sqrt(n)
        sub_pos   = nx.spring_layout(subg, seed=seed + rank, weight="weight",
                                     k=k_param, iterations=80)
        for node, (x, y) in sub_pos.items():
            pos[node] = (cx + x * mod_scale, cy + y * mod_scale)

    #scatter small components at random angles around the periphery
    if pos and small:
        all_xs = [x for x, y in pos.values()]
        all_ys = [y for x, y in pos.values()]
        r_main = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) / 2 * 1.3
        cx_main = (max(all_xs) + min(all_xs)) / 2
        cy_main = (max(all_ys) + min(all_ys)) / 2
        angles = rng.uniform(0, 2 * math.pi, size=len(small))
        radii  = rng.uniform(r_main, r_main * 1.5, size=len(small))
        for node, angle, r in zip(small, angles, radii):
            pos[node] = (cx_main + r * math.cos(angle), cy_main + r * math.sin(angle))
    elif small:
        for node in small:
            pos[node] = (rng.uniform(-5, 5), rng.uniform(-5, 5))

    return pos


_TOP_MODS_COLORED = 15   # top N Leiden modules get a vivid colour; rest get gray


def draw_network(
    g: nx.Graph,
    title: str,
    subtitle: str,
    edge_type: str,          #"C", "D", or "S" — determines edge colour
    out_path: Path,
    dpi: int,
    seed: int,
    edge_alpha: float = 0.25,
    edge_width: float = 0.7,
    edge_quantile: float = 0.30,
    figsize: tuple[float, float] = (22, 18),
) -> None:
    #draw single-edge-type network coloured by leiden modules
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No edges to display",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG_COLOR)
        plt.close(fig)
        return

    #remove components < 8 genes so only biologically deep clusters are shown
    MIN_COMP = 8
    keep = {n for comp in nx.connected_components(g) if len(comp) >= MIN_COMP for n in comp}
    n_excluded = g.number_of_nodes() - len(keep)
    g = g.subgraph(keep).copy()

    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, f"No components with ≥ {MIN_COMP} nodes",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG_COLOR)
        plt.close(fig)
        return

    #leiden clustering used for both layout and colouring
    mod_map = leiden_on_graph(g, seed=seed)
    pos     = community_layout(g, mod_map, seed=seed)

    #rank modules by size; top _TOP_MODS_COLORED get vivid colours, rest gray
    mod_sizes: dict[int, int] = {}
    for n_, m in mod_map.items():
        mod_sizes[m] = mod_sizes.get(m, 0) + 1
    real_mods = sorted([m for m in mod_sizes if m != 0],
                       key=lambda m: mod_sizes[m], reverse=True)
    n_total_mods = len(real_mods)
    top_mods = real_mods[:_TOP_MODS_COLORED]
    palette  = _make_palette(len(top_mods))
    mod_color: dict[int, str] = {m: palette[i] for i, m in enumerate(top_mods)}

    node_color_map = {
        n_: mod_color.get(mod_map.get(n_, 0), _SINGLETON_COLOR)
        for n_ in g.nodes()
    }

    #draw top-quantile edges
    ec_hex = EDGE_COLORS[edge_type]
    all_e  = list(g.edges(data=True))
    elist  = []
    if all_e:
        weights   = np.array([d.get("weight", 0.0) for _, _, d in all_e], dtype=float)
        threshold = float(np.quantile(weights, edge_quantile))
        elist = [(u, v) for (u, v, d), w in zip(all_e, weights) if w >= threshold]
        if elist:
            nx.draw_networkx_edges(g, pos, edgelist=elist, ax=ax,
                                   edge_color=ec_hex, alpha=edge_alpha, width=edge_width)

    #draw nodes sized by weighted degree
    sizes     = degree_scaled_sizes(g)
    node_list = list(g.nodes())
    node_colors = [node_color_map[n_] for n_ in node_list]
    node_sizes  = [sizes.get(n_, _SINGLETON_SIZE) for n_ in node_list]

    nx.draw_networkx_nodes(g, pos, nodelist=node_list, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.88, linewidths=0.3, edgecolors="#ffffff")

    hubs = top_hub_genes(g, TOP_LABEL_GENES)
    if hubs:
        nx.draw_networkx_labels(g, pos, labels={n_: n_ for n_ in hubs}, ax=ax,
                                font_size=9, font_color="#1d3557", font_weight="bold")

    #legend: vivid colours for top modules, gray for rest
    patches = [mpatches.Patch(color=mod_color[m],
                               label=f"Module {i+1}  (n={mod_sizes[m]})")
               for i, m in enumerate(top_mods)]
    n_gray = n_total_mods - len(top_mods)
    if n_gray > 0:
        patches.append(mpatches.Patch(color=_SINGLETON_COLOR,
                                       label=f"Modules {len(top_mods)+1}–{n_total_mods}  (smaller)"))
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.75,
              edgecolor="none", title=f"{n_total_mods} Leiden modules total",
              title_fontsize=8)

    ep = mpatches.Patch(color=ec_hex, label=EDGE_LABELS[edge_type])
    fig.legend(handles=[ep], loc="lower center", fontsize=10,
               framealpha=0.8, edgecolor="none", bbox_to_anchor=(0.5, -0.02))

    n_shown = len(elist)
    excl_note = f" · {n_excluded:,} nodes in small components (< {MIN_COMP} genes) excluded" if n_excluded > 0 else ""
    ax.set_title(
        f"{title}\n{subtitle}\n"
        f"{g.number_of_nodes():,} nodes · {n_total_mods} Leiden modules · "
        f"{n_shown:,} / {g.number_of_edges():,} edges shown (top {int((1-edge_quantile)*100)}%)"
        f"{excl_note}",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)


def draw_shared_network(
    g: nx.Graph,
    title: str,
    subtitle: str,
    out_path: Path,
    dpi: int,
    seed: int,
    edge_alpha: float = 0.20,
    edge_quantile: float = 0.40,
    figsize: tuple[float, float] = (22, 18),
) -> None:
    #draw shared network with all three csd edge types
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG_COLOR)
        plt.close(fig)
        return

    mod_map = leiden_on_graph(g, seed=seed)
    mod_sizes: dict[int, int] = {}
    for n_, m in mod_map.items():
        mod_sizes[m] = mod_sizes.get(m, 0) + 1
    real_mods = sorted([m for m in mod_sizes if m != 0], key=lambda m: mod_sizes[m], reverse=True)
    palette   = _make_palette(len(real_mods))
    mod_color = {m: palette[i] for i, m in enumerate(real_mods)}

    pos = community_layout(g, mod_map, seed=seed)

    #draw top-quantile edges, one pass per csd type
    all_e   = list(g.edges(data=True))
    weights = np.array([d.get("weight", 0.0) for _, _, d in all_e], dtype=float)
    thresh  = float(np.quantile(weights, edge_quantile))
    n_drawn = 0
    for csd_type, ec in EDGE_COLORS.items():
        elist = [(u, v) for (u, v, d), w in zip(all_e, weights)
                 if d.get("link_type") == csd_type and w >= thresh]
        if elist:
            nx.draw_networkx_edges(g, pos, edgelist=elist, ax=ax,
                                   edge_color=ec, alpha=edge_alpha, width=0.6)
            n_drawn += len(elist)

    sizes      = degree_scaled_sizes(g)
    node_list  = list(g.nodes())
    node_colors = [mod_color.get(mod_map.get(n_, 0), _SINGLETON_COLOR) for n_ in node_list]
    node_sizes  = [sizes.get(n_, _SINGLETON_SIZE) for n_ in node_list]

    nx.draw_networkx_nodes(g, pos, nodelist=node_list, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.88, linewidths=0.3, edgecolors="#ffffff")

    hubs = top_hub_genes(g, TOP_LABEL_GENES)
    if hubs:
        nx.draw_networkx_labels(g, pos, labels={n_: n_ for n_ in hubs}, ax=ax,
                                font_size=9, font_color="#1d3557", font_weight="bold")

    #module legend
    legend_mods = real_mods[:12]
    patches = [mpatches.Patch(color=mod_color[m], label=f"Module {i+1}  (n={mod_sizes[m]})")
               for i, m in enumerate(legend_mods)]
    if len(real_mods) > 12:
        patches.append(mpatches.Patch(color="#cccccc", label=f"+ {len(real_mods)-12} more modules"))
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.75,
              edgecolor="none", title=f"{len(real_mods)} modules", title_fontsize=8)

    #csd edge legend
    edge_patches = [mpatches.Patch(color=EDGE_COLORS[t], label=EDGE_LABELS[t])
                    for t in ("C", "S", "D")]
    fig.legend(handles=edge_patches, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.8, edgecolor="none", bbox_to_anchor=(0.5, -0.02))

    n_total = g.number_of_edges()
    ax.set_title(
        f"{title}\n{subtitle}\n"
        f"{g.number_of_nodes():,} nodes · {len(real_mods)} modules · "
        f"{n_drawn:,} / {n_total:,} edges shown (top {int((1-edge_quantile)*100)}%)",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)


#per-pair runner
def run_pair(pair_name: str, network_dir: Path, shared_dir: Path,
             out_root: Path, dpi: int, seed: int) -> None:
    if "__vs__" not in pair_name:
        raise ValueError(f"Invalid pair name: {pair_name}")
    case, ctrl = pair_name.split("__vs__", 1)

    edges_file    = network_dir / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    presence_file = shared_dir  / pair_name / f"{pair_name}_gene_presence.csv"
    if not edges_file.exists():
        print(f"  [SKIP] {edges_file}")
        return
    if not presence_file.exists():
        print(f"  [SKIP] {presence_file}")
        return

    print(f"[{pair_name}]")
    edges_df = load_edges(edges_file)
    pres_df  = pd.read_csv(presence_file)
    pres_map = dict(zip(pres_df["gene"].astype(str), pres_df["presence"].astype(str)))
    shared_genes = {g for g, p in pres_map.items() if p == "shared"}

    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    label = pair_name.replace("__vs__", " vs ")

    #fig 1: shared genes, all C/S/D
    print("  [1/5] shared network (C+S+D)…")
    g_shared = build_graph(edges_df, {"C", "S", "D"}, node_filter=shared_genes)
    print(f"        {g_shared.number_of_nodes():,} nodes  {g_shared.number_of_edges():,} edges")
    draw_shared_network(
        g_shared,
        title="Shared-gene co-expression network",
        subtitle=label,
        out_path=pair_out / f"{pair_name}_shared_network.png",
        dpi=dpi, seed=seed,
    )

    #fig 2: shared genes, C only
    print("  [2/5] shared C edges…")
    g_c = build_graph(edges_df, {"C"}, node_filter=shared_genes)
    print(f"        {g_c.number_of_nodes():,} nodes  {g_c.number_of_edges():,} edges")
    draw_network(
        g_c,
        title="Conserved co-expression (C edges) — shared genes",
        subtitle=f"{label}  |  Co-expression maintained across tumour and normal",
        edge_type="C",
        out_path=pair_out / f"{pair_name}_shared_C.png",
        dpi=dpi, seed=seed, edge_alpha=0.25, edge_quantile=0.35,
        figsize=(28, 20),
    )

    #fig 3: shared genes, D only
    print("  [3/5] shared D edges…")
    g_d = build_graph(edges_df, {"D"}, node_filter=shared_genes)
    print(f"        {g_d.number_of_nodes():,} nodes  {g_d.number_of_edges():,} edges")
    draw_network(
        g_d,
        title="Reprogrammed co-expression (D edges) — shared genes",
        subtitle=f"{label}  |  Co-expression relationship altered in cancer",
        edge_type="D",
        out_path=pair_out / f"{pair_name}_shared_D.png",
        dpi=dpi, seed=seed, edge_alpha=0.30, edge_quantile=0.30,
        figsize=(28, 20),
    )

    #fig 4: case-specific S edges, all genes
    print("  [4/5] tumour-gained S edges…")
    g_s_case = build_directional_s_graph(edges_df, case_dominant=True)
    print(f"        {g_s_case.number_of_nodes():,} nodes  {g_s_case.number_of_edges():,} edges")
    draw_network(
        g_s_case,
        title=f"Tumour-gained co-expression (S edges, case-dominant)",
        subtitle=f"{label}  |  Co-expression present in {case}, near-zero in {ctrl}",
        edge_type="S",
        out_path=pair_out / f"{pair_name}_specific_S_case.png",
        dpi=dpi, seed=seed, edge_alpha=0.30, edge_quantile=0.30,
    )

    #fig 5: ctrl-specific S edges, all genes
    print("  [5/5] normal-retained S edges…")
    g_s_ctrl = build_directional_s_graph(edges_df, case_dominant=False)
    print(f"        {g_s_ctrl.number_of_nodes():,} nodes  {g_s_ctrl.number_of_edges():,} edges")
    draw_network(
        g_s_ctrl,
        title=f"Normal-retained co-expression (S edges, ctrl-dominant)",
        subtitle=f"{label}  |  Co-expression present in {ctrl}, near-zero in {case}",
        edge_type="S",
        out_path=pair_out / f"{pair_name}_specific_S_ctrl.png",
        dpi=dpi, seed=seed, edge_alpha=0.30, edge_quantile=0.30,
    )

    print(f"  done → {pair_out}")


#cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSD panel figures for union networks")
    p.add_argument("--network-dir", default="results/14_csd_networks")
    p.add_argument("--shared-dir",  default="results/12_csd_scoring")
    p.add_argument("--output-dir",  default="results/16_network_viz")
    p.add_argument("--pair", action="append", default=[])
    p.add_argument("--dpi",  type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> int:
    args    = parse_args()

    #resolve paths and create output dir
    net_dir = resolve_base(args.network_dir)
    shr_dir = resolve_base(args.shared_dir)
    out_dir = resolve_base(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #discover pair dirs and filter to requested subset if specified
    pair_dirs = sorted([p for p in net_dir.iterdir() if p.is_dir() and "__vs__" in p.name])
    if args.pair:
        keep = set(args.pair)
        pair_dirs = [p for p in pair_dirs if p.name in keep]
    if not pair_dirs:
        raise ValueError("No pair directories found")

    #generate five panel figures per pair
    for pair_dir in pair_dirs:
        run_pair(pair_name=pair_dir.name, network_dir=net_dir, shared_dir=shr_dir,
                 out_root=out_dir, dpi=args.dpi, seed=args.seed)

    print(f"\nDone → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
