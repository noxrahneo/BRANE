#!/usr/bin/env python3
"""
Script 47: Prepare visualization assets for 20_node_annotation.

For each CSD network pair assembles annotated node tables, tier-specific PNGs,
and an interactive HTML network from template.html.

INPUTS (per pair):
  14_csd_networks/{pair}/
    {pair}_node_homogeneity_permutation.csv  -- topology + DEG stats for all nodes
    {pair}_differential_edges_permutation.csv -- edges with link_type (C/D/S)
    {pair}_leiden_modules.tsv                -- module assignments
    {pair}_top_hubs_permutation.csv          -- top 50 hub genes overall
  20_node_annotation/03_output_with_lfc/{pair}_tagged_with_lfc.csv

OUTPUTS (per pair) in 20_node_annotation/{pair}/:
  nodes.csv                  -- topology + DEG + cell type + cancer gene
  modules.csv                -- module-level summary
  network_summary.csv        -- key network statistics
  hubs_overall.csv           -- top hubs, fully annotated
  hubs_D.csv                 -- hubs within D-tier subnetwork
  hubs_S_case.csv            -- hubs within S_case-tier subnetwork
  hubs_S_ctrl.csv            -- hubs within S_ctrl-tier subnetwork
  network_annotated.html     -- interactive vis-network (template.html-based)
  figures/
    full_annotated.png       -- all edges, nodes coloured by module
    tier_CD.png              -- C + D edges (community layout)
    tier_D.png               -- D edges only (community layout)
    tier_S_case.png          -- S_case edges only (community layout)
    tier_S_ctrl.png          -- S_ctrl edges only (community layout)
    modules_case.png         -- case co-expression (C + S_case), annotated
    modules_ctrl.png         -- normal co-expression (C + S_ctrl), annotated
    modules_shared.png       -- conserved co-expression (C only), annotated
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "results" / "20_node_annotation"
CSD_DIR = REPO_ROOT / "results" / "14_csd_networks"
LFC_DIR = BASE / "03_output_with_lfc"
TEMPLATE_HTML = Path(__file__).parent / "template.html"

PAIRS = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

# Short prefix added to every output file inside the pair folder, so files are
# self-identifying even when viewed or copied outside their parent directory.
PAIR_SHORT = {
    "ER_tumor__vs__Normal":                                           "ER",
    "HER2_tumor__vs__Normal":                                         "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal":                      "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal":                        "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal":                              "TNBC",
}

TOP_HUBS = 50
TOP_LABEL_N = 40
MIN_MOD_SIZE_FULL = 15   # minimum module size to display in full/CD/modules views
MIN_MOD_SIZE_TIER = 5    # relaxed threshold for tier-specific views

# Edge colors and labels for legend — S_case and S_ctrl share the same visual S color
EDGE_COLORS = {
    "C": "#2b6fb0",
    "D": "#e63946",
    "S": "#2a9d8f",   # visual type for both S_case and S_ctrl
}
EDGE_LABELS = {
    "C": "Conserved (C)",
    "D": "Differential / Rewired (D)",
    "S": "Condition-specific (S)",
}

CELL_TYPE_BORDER_COLORS = {
    "bcell": "#0072B2", "cycling": "#E74C3C", "epithelial": "#2ECC71",
    "endo": "#F39C12", "endothelial": "#F39C12", "fibroblast": "#6D4C41",
    "fibro": "#00ACC1", "myeloid": "#9B59B6", "tcell": "#3498DB",
    "t_cell": "#3498DB", "luminal_epi": "#27AE60", "luminal": "#27AE60",
    "basal_epi": "#16A085", "basal": "#16A085", "plasma": "#E91E63",
    "nk": "#F1C40F", "unknown": "#9CA3AF",
}
CT_DISPLAY_NAMES = {
    "bcell": "B cell", "cycling": "Cycling", "epithelial": "Epithelial",
    "endo": "Endothelial", "endothelial": "Endothelial",
    "fibroblast": "Fibroblast", "fibro": "Fibroblast",
    "myeloid": "Myeloid", "tcell": "T cell", "t_cell": "T cell",
    "luminal_epi": "Luminal epithelial", "luminal": "Luminal",
    "basal_epi": "Basal epithelial", "basal": "Basal",
    "plasma": "Plasma", "nk": "NK cell", "unknown": "Unknown",
}

_MODULE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d3d3d3", "#888888",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cell_type_border_color(ct: str | None) -> str:
    if not ct or str(ct).lower() in {"nan", "none", ""}:
        return "#9CA3AF"
    return CELL_TYPE_BORDER_COLORS.get(str(ct).lower().replace(" ", "_"), "#9CA3AF")


def ct_display_name(ct: str | None) -> str:
    if not ct or str(ct).lower() in {"nan", "none", ""}:
        return "Unknown"
    key = str(ct).lower().replace(" ", "_")
    return CT_DISPLAY_NAMES.get(key, str(ct).replace("_", " ").title())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_edges(pair: str) -> pd.DataFrame:
    path = CSD_DIR / pair / f"{pair}_differential_edges_permutation.csv"
    df = pd.read_csv(path)
    def _detail(row: pd.Series) -> str:
        if row["link_type"] != "S":
            return str(row["link_type"])
        return "S_case" if float(row.get("rho_case", 0)) > float(row.get("rho_control", 0)) else "S_ctrl"
    df["link_type_detail"] = df.apply(_detail, axis=1)
    return df


def load_modules(pair: str) -> dict[str, int]:
    path = CSD_DIR / pair / f"{pair}_leiden_modules.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if "gene" not in df.columns or "module" not in df.columns:
        return {}
    return {str(r["gene"]): int(r["module"]) for _, r in df.iterrows() if str(r["gene"]).strip()}


def load_homogeneity(pair: str) -> pd.DataFrame:
    path = CSD_DIR / pair / f"{pair}_node_homogeneity_permutation.csv"
    return pd.read_csv(path)


def load_annotation(pair: str) -> pd.DataFrame:
    path = LFC_DIR / f"{pair}_tagged_with_lfc.csv"
    if not path.exists():
        logging.warning("Annotation file missing: %s — annotation columns will be empty", path.name)
        return pd.DataFrame(columns=["gene", "approved_symbol"])
    keep = [
        "gene", "approved_symbol", "hgnc_id", "entrez_id", "full_name",
        "known_cancer_gene", "cancer_role", "evidence_tier",
        "oncokb_is_oncogene", "oncokb_is_tsg",
        "cell_type", "major_compartment", "ct_switched",
        "lfc", "direction",
    ]
    df = pd.read_csv(path)
    present = [c for c in keep if c in df.columns]
    return df[present].copy()


def load_hubs_raw(pair: str) -> pd.DataFrame:
    path = CSD_DIR / pair / f"{pair}_top_hubs_permutation.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Nodes, modules, hubs, summary assembly
# ---------------------------------------------------------------------------

def build_nodes_df(pair: str) -> pd.DataFrame:
    homo = load_homogeneity(pair)
    ann = load_annotation(pair)

    ann_key = "approved_symbol" if "approved_symbol" in ann.columns else "gene"
    homo_key = "gene"

    df = homo.merge(ann, left_on=homo_key, right_on=ann_key, how="left", suffixes=("", "_ann"))

    if "lfc" in df.columns:
        df = df.rename(columns={"lfc": "expr_lfc", "direction": "expr_direction"})
    if "gene_ann" in df.columns:
        df = df.drop(columns=["gene_ann"])

    front = [
        "gene", "approved_symbol", "hgnc_id", "entrez_id", "full_name",
        "degree", "weighted_degree", "clustering_coefficient",
        "closeness_centrality", "betweenness_centrality",
        "module", "avg_wTO",
        "lnFC", "fdr", "is_deg", "deg_direction",
        "expr_lfc", "expr_direction",
        "cell_type", "major_compartment",
        "known_cancer_gene", "cancer_role", "evidence_tier",
        "oncokb_is_oncogene", "oncokb_is_tsg",
    ]
    present = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    return df[present + rest].copy()


def build_modules_df(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    if "module" not in nodes_df.columns:
        return pd.DataFrame()
    rows = []
    for mod_id, grp in nodes_df.groupby("module"):
        genes = grp["gene"].tolist()
        gene_set = set(genes)
        sub_edges = edges_df[
            edges_df["gene_a"].isin(gene_set) & edges_df["gene_b"].isin(gene_set)
        ]
        top_by_degree = (
            grp.nlargest(5, "degree")["gene"].tolist()
            if "degree" in grp.columns else []
        )
        rows.append({
            "module": int(mod_id),
            "n_genes": len(genes),
            "n_edges": len(sub_edges),
            "n_deg_genes": int(grp["is_deg"].sum()) if "is_deg" in grp.columns else 0,
            "top_genes_by_degree": "|".join(top_by_degree),
            "dominant_cell_type": (
                grp["cell_type"].mode().iloc[0]
                if "cell_type" in grp.columns and grp["cell_type"].notna().any() else ""
            ),
        })
    return pd.DataFrame(rows).sort_values("n_genes", ascending=False)


def build_summary_stats(
    pair: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    modules_df: pd.DataFrame,
) -> pd.DataFrame:
    ltype = edges_df["link_type_detail"] if "link_type_detail" in edges_df.columns else pd.Series(dtype=str)
    row = {
        "pair": pair.replace("__vs__", " vs "),
        "n_nodes": len(nodes_df),
        "n_edges_total": len(edges_df),
        "n_edges_C": int((ltype == "C").sum()),
        "n_edges_D": int((ltype == "D").sum()),
        "n_edges_S_case": int((ltype == "S_case").sum()),
        "n_edges_S_ctrl": int((ltype == "S_ctrl").sum()),
        "n_modules": len(modules_df),
        "n_deg_genes": int(nodes_df["is_deg"].sum()) if "is_deg" in nodes_df.columns else 0,
        "n_up_genes": int((nodes_df.get("deg_direction", pd.Series()) == "up").sum()),
        "n_down_genes": int((nodes_df.get("deg_direction", pd.Series()) == "down").sum()),
        "n_cancer_genes": int(nodes_df["known_cancer_gene"].sum()) if "known_cancer_gene" in nodes_df.columns else 0,
        "n_oncogenes": int((nodes_df.get("cancer_role", pd.Series()).str.lower() == "oncogene").sum()),
        "n_tsgs": int((nodes_df.get("cancer_role", pd.Series()).str.lower() == "tsg").sum()),
        "pct_cell_type_assigned": (
            round(100.0 * nodes_df["cell_type"].notna().sum() / len(nodes_df), 2)
            if "cell_type" in nodes_df.columns and len(nodes_df) > 0 else 0.0
        ),
    }
    return pd.DataFrame([row])


def _tier_hubs(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    tier: str,
    n: int = TOP_HUBS,
) -> pd.DataFrame:
    sub = edges_df[edges_df["link_type_detail"] == tier]
    if sub.empty:
        return pd.DataFrame()

    g = nx.Graph()
    for _, r in sub.iterrows():
        g.add_edge(str(r["gene_a"]), str(r["gene_b"]),
                   weight=float(r.get("weight", 1.0)))

    deg = dict(g.degree(weight="weight"))
    sorted_genes = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:n]

    ann_cols = ["approved_symbol", "known_cancer_gene", "cancer_role",
                "evidence_tier", "cell_type", "major_compartment", "lnFC",
                "is_deg", "deg_direction", "module"]
    present = [c for c in ann_cols if c in nodes_df.columns]
    ann_map = nodes_df.set_index("gene")[present].to_dict("index")

    rows = []
    for rank, (gene, tier_degree) in enumerate(sorted_genes, start=1):
        row: dict[str, Any] = {"rank": rank, "gene": gene, "tier_degree": tier_degree}
        row.update(ann_map.get(gene, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def build_hubs_overall(pair: str, nodes_df: pd.DataFrame) -> pd.DataFrame:
    raw = load_hubs_raw(pair)
    if raw.empty or nodes_df.empty:
        return raw
    ann_cols = ["approved_symbol", "known_cancer_gene", "cancer_role",
                "evidence_tier", "cell_type", "major_compartment", "lnFC",
                "is_deg", "deg_direction", "module"]
    present = [c for c in ann_cols if c in nodes_df.columns]
    ann_map = nodes_df.set_index("gene")[present].to_dict("index")
    rows = []
    for _, r in raw.iterrows():
        gene = str(r["gene"])
        row = r.to_dict()
        row.update(ann_map.get(gene, {}))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layout and drawing utilities
# ---------------------------------------------------------------------------

def module_color(mod_id: int | None) -> str:
    if mod_id is None:
        return "#cccccc"
    return _MODULE_PALETTE[(int(mod_id) - 1) % len(_MODULE_PALETTE)]


def degree_scaled_sizes(g: nx.Graph, lo: float = 20.0, hi: float = 220.0) -> dict[str, float]:
    dw = dict(g.degree(weight="weight"))
    vals = np.array(list(dw.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        return {n: lo + (hi - lo) * (v - vmin) / (vmax - vmin) for n, v in dw.items()}
    return {n: (lo + hi) / 2 for n in dw}


def top_hub_genes(g: nx.Graph, n: int) -> list[str]:
    dw = dict(g.degree(weight="weight"))
    return sorted(dw, key=lambda x: dw[x], reverse=True)[:n]


def community_layout(g: nx.Graph, node_to_module: dict[str, int], seed: int = 7) -> dict[str, tuple[float, float]]:
    """Communities arranged via spring on a meta-graph; spring within each community.
    Copied from script 32 (same algorithm, same seed behaviour).
    """
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
        return dict(nx.spring_layout(g.subgraph(nodes), seed=seed, weight="weight", iterations=80))

    node_to_mod = {n: m for m, nodes in mods for n in nodes}
    meta = nx.Graph()
    for m, _ in mods:
        meta.add_node(m)
    for u, v, d in g.edges(data=True):
        mu, mv = node_to_mod.get(u), node_to_mod.get(v)
        if mu is not None and mv is not None and mu != mv:
            w = d.get("weight", 1.0)
            if meta.has_edge(mu, mv):
                meta[mu][mv]["weight"] += w
            else:
                meta.add_edge(mu, mv, weight=w)

    spread = max(100.0, n_mods * 18.0)
    k_meta = spread / math.sqrt(n_mods)
    meta_pos_raw = nx.spring_layout(meta, seed=seed, weight="weight", k=k_meta, iterations=150)
    xs = [x for x, _ in meta_pos_raw.values()]
    ys = [y for _, y in meta_pos_raw.values()]
    x_range, y_range = max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0
    scale = spread / max(x_range, y_range)
    centres = {m: (x * scale, y * scale) for m, (x, y) in meta_pos_raw.items()}
    max_mod_scale = spread / (1.5 * math.sqrt(n_mods))

    pos: dict[str, tuple[float, float]] = {}
    for local_idx, (mod_id, nodes) in enumerate(mods):
        cx, cy = centres[mod_id]
        n = len(nodes)
        if n == 1:
            pos[nodes[0]] = (cx, cy)
        elif n == 2:
            pos[nodes[0]], pos[nodes[1]] = (cx - 0.5, cy), (cx + 0.5, cy)
        else:
            k_param = 18.0 / math.sqrt(n)
            module_scale = min(max_mod_scale, max(4.0, math.sqrt(n) * 2.0))
            sub_pos = nx.spring_layout(
                g.subgraph(nodes), seed=seed + local_idx, weight="weight",
                k=k_param, iterations=80,
            )
            for node, (x, y) in sub_pos.items():
                pos[node] = (cx + x * module_scale, cy + y * module_scale)
    return pos


def _build_graph_from_edges(
    edges_df: pd.DataFrame,
    tier_filter: list[str] | None = None,
) -> nx.Graph:
    if tier_filter is not None:
        edges_df = edges_df[edges_df["link_type_detail"].isin(tier_filter)]
    g = nx.Graph()
    for _, r in edges_df.iterrows():
        a, b = str(r["gene_a"]), str(r["gene_b"])
        if not a or not b:
            continue
        w = float(r.get("weight", 1.0)) if pd.notna(r.get("weight")) else 1.0
        lt = str(r.get("link_type_detail", "C"))
        # Collapse S_case/S_ctrl to "S" for visual edge colouring (same green)
        lt_visual = "S" if lt.startswith("S") else lt
        g.add_edge(a, b, weight=w, link_type=lt_visual)
    return g


def _build_legends(
    fig: plt.Figure,
    node_list: list[str],
    node_info: dict[str, dict],
    modules_map: dict[str, int],
    mod_sizes: dict[int, int],
    edge_types_present: set[str],
) -> None:
    """Build a single merged legend panel on the right side of the figure."""

    all_handles: list = []
    all_labels: list[str] = []

    # --- Section 1: Edge types ---
    all_handles.append(mpatches.Patch(color="none", label="── Edge types ──"))
    all_labels.append("── Edge types ──")
    for t in ["C", "D", "S"]:
        if t in edge_types_present:
            all_handles.append(
                Line2D([0], [0], color=EDGE_COLORS[t], linewidth=2.5, label=EDGE_LABELS[t])
            )
            all_labels.append(EDGE_LABELS[t])

    # --- Section 2: Cell type (node border) ---
    present_cts: dict[str, str] = {}
    for n in node_list:
        ct_raw = node_info.get(n, {}).get("cell_type")
        if ct_raw and str(ct_raw).lower() not in {"nan", "none", ""}:
            key = str(ct_raw).lower().replace(" ", "_")
            present_cts[key] = ct_display_name(ct_raw)

    if present_cts:
        all_handles.append(mpatches.Patch(color="none", label="── Cell type (border) ──"))
        all_labels.append("── Cell type (border) ──")
        for key, display in sorted(present_cts.items(), key=lambda x: x[1])[:12]:
            h = Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="none",
                markeredgecolor=CELL_TYPE_BORDER_COLORS.get(key, "#9CA3AF"),
                markeredgewidth=2.5, markersize=11,
                label=display,
            )
            all_handles.append(h)
            all_labels.append(display)

    # --- Section 3: Leiden modules (top 12) ---
    top_mods = sorted(mod_sizes.items(), key=lambda x: x[1], reverse=True)[:12]
    if top_mods:
        all_handles.append(mpatches.Patch(color="none", label="── Leiden modules ──"))
        all_labels.append("── Leiden modules ──")
        for mid, sz in top_mods:
            h = Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=module_color(mid),
                markeredgecolor="#555555", markeredgewidth=0.5,
                markersize=11,
                label=f"Module {mid}  ({sz} genes)",
            )
            all_handles.append(h)
            all_labels.append(f"Module {mid}  ({sz} genes)")

    # --- Section 4: Node annotation symbols ---
    all_handles.append(mpatches.Patch(color="none", label="── Node annotation ──"))
    all_labels.append("── Node annotation ──")
    for lbl in [
        "★  Oncogene",
        "▲  Tumour suppressor (TSG)",
        "↑  Upregulated DEG",
        "↓  Downregulated DEG",
        "−  Stable expression",
        "bold = hub gene (top degree)",
    ]:
        all_handles.append(mpatches.Patch(color="none", label=lbl))
        all_labels.append(lbl)

    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        borderpad=1.0,
        labelspacing=0.55,
        handlelength=1.8,
        ncol=4,
    )


def draw_annotated_network(
    g: nx.Graph,
    out_png: Path,
    title: str,
    nodes_df: pd.DataFrame,
    modules_map: dict[str, int],
    seed: int = 7,
    use_community_layout: bool = True,
    min_mod_size: int = MIN_MOD_SIZE_FULL,
) -> None:
    """Draw CSD network: module fill + cell type borders + cancer/DEG labels + 4 legends."""
    if g.number_of_nodes() == 0:
        return

    node_info: dict[str, dict] = {}
    if nodes_df is not None:
        for _, r in nodes_df.iterrows():
            gene = str(r.get("gene", ""))
            if gene:
                node_info[gene] = r.to_dict()

    # Filter to large modules (same as script 32)
    mod_sizes: dict[int, int] = {}
    for n in g.nodes():
        mid = modules_map.get(n)
        if mid is not None:
            mod_sizes[mid] = mod_sizes.get(mid, 0) + 1

    if use_community_layout:
        large_mods = {mid for mid, sz in mod_sizes.items() if sz >= min_mod_size}
        display_nodes = {n for n in g.nodes() if modules_map.get(n) in large_mods}
        g = g.subgraph(display_nodes).copy()
        small = [n for comp in nx.connected_components(g) if len(comp) <= 2 for n in comp]
        if small:
            g = g.copy()
            g.remove_nodes_from(small)

    if g.number_of_nodes() == 0:
        return

    # Recalculate mod_sizes after filter for legend
    mod_sizes_display: dict[int, int] = {}
    for n in g.nodes():
        mid = modules_map.get(n)
        if mid is not None:
            mod_sizes_display[mid] = mod_sizes_display.get(mid, 0) + 1

    node_list = list(g.nodes())
    pos = (
        community_layout(g, modules_map, seed=seed)
        if use_community_layout
        else nx.spring_layout(g, seed=seed, weight="weight", iterations=100)
    )

    sizes = degree_scaled_sizes(g, lo=5.0, hi=70.0)
    fills = [module_color(modules_map.get(n)) for n in node_list]
    borders = [cell_type_border_color(node_info.get(n, {}).get("cell_type")) for n in node_list]
    sz = [sizes.get(n, 25.0) for n in node_list]

    fig, ax = plt.subplots(1, 1, figsize=(14, 20))

    # Edges by CSD type
    edge_types_present: set[str] = set()
    for csd_type, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, d in g.edges(data=True) if d.get("link_type") == csd_type]
        if elist:
            edge_types_present.add(csd_type)
            nx.draw_networkx_edges(g, pos, edgelist=elist, ax=ax,
                                   edge_color=color, alpha=0.20, width=0.4)

    nx.draw_networkx_nodes(g, pos, nodelist=node_list, ax=ax,
                           node_color=fills, node_size=sz, alpha=0.88,
                           linewidths=1.8, edgecolors=borders)

    # Labels: top hub genes, bold, with cancer/DEG symbols — adjustText for overlap
    hubs = set(top_hub_genes(g, TOP_LABEL_N))
    cancer_sym = {"oncogene": "★", "tsg": "▲", "Oncogene": "★", "TSG": "▲"}
    dir_sym = {"up": "↑", "down": "↓", "unchanged": "−"}
    texts = []
    for n in node_list:
        if n not in hubs:
            continue
        info = node_info.get(n, {})
        csym = cancer_sym.get(str(info.get("cancer_role", "")), "")
        dsym = dir_sym.get(str(info.get("deg_direction", "")).lower(), "")
        lbl = f"{n}{csym}{dsym}"
        x, y = pos[n]
        texts.append(ax.text(x, y, lbl, fontsize=6.5, color="#1d3557",
                             fontweight="bold", ha="center", va="center"))
    if texts:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
                expand_text=(1.2, 1.4),
                expand_points=(1.1, 1.2),
                force_text=(0.4, 0.6),
                force_points=(0.2, 0.3),
            )

    # Tighten axes to actual node positions — removes dead space around clusters
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    mx = (max(xs) - min(xs)) * 0.01
    my = (max(ys) - min(ys)) * 0.01
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.18)
    _build_legends(fig, node_list, node_info, modules_map, mod_sizes_display, edge_types_present)

    ax.set_title(title, fontsize=16, pad=8, fontweight="bold")
    ax.axis("off")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    logging.info("  %s (%d nodes, %d edges)", out_png.name, g.number_of_nodes(), g.number_of_edges())


def draw_top6_module_panels(
    pair: str,
    tier: str,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    modules_map: dict[str, int],
    out_path: Path,
    hub_genes: set[str] | None = None,
) -> None:
    """3×2 grid showing the top 6 modules from the tier subnetwork, one subplot each."""
    g_tier = _build_graph_from_edges(edges_df, tier_filter=[tier])
    if g_tier.number_of_nodes() == 0:
        logging.warning("  %s tier %s: empty — skipping module panels", pair, tier)
        return

    # Count how many tier-graph nodes fall in each module
    mod_sizes_in_tier: dict[int, int] = {}
    for n in g_tier.nodes():
        mid = modules_map.get(n)
        if mid is not None:
            mod_sizes_in_tier[mid] = mod_sizes_in_tier.get(mid, 0) + 1

    if not mod_sizes_in_tier:
        logging.warning("  %s tier %s: no module assignments — skipping", pair, tier)
        return

    # Exclude singletons and doubletons — only modules with ≥ 3 nodes
    top6 = sorted(
        [(mid, sz) for mid, sz in mod_sizes_in_tier.items() if sz >= 3],
        key=lambda x: x[1], reverse=True
    )[:6]

    node_info: dict[str, dict] = {}
    for _, r in nodes_df.iterrows():
        gene = str(r.get("gene", ""))
        if gene:
            node_info[gene] = r.to_dict()

    tier_color = EDGE_COLORS.get("S" if tier.startswith("S") else tier, "#888888")
    tier_display = {
        "D": "Rewired (D)",
        "S_case": "Tumour-gained (S+)",
        "S_ctrl": "Normal-lost (S−)",
    }
    cancer_sym = {"oncogene": "★", "tsg": "▲", "Oncogene": "★", "TSG": "▲"}
    dir_sym = {"up": "↑", "down": "↓", "unchanged": "−"}

    fig, axes = plt.subplots(3, 2, figsize=(14, 20))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14, hspace=0.38, wspace=0.10)
    axes_flat = axes.flatten()

    all_ct_keys: dict[str, str] = {}
    mod_legend_handles: list = []

    for panel_idx in range(6):
        ax = axes_flat[panel_idx]
        if panel_idx >= len(top6):
            ax.axis("off")
            continue

        mod_id, mod_size = top6[panel_idx]
        mod_nodes = [n for n in g_tier.nodes() if modules_map.get(n) == mod_id]
        if not mod_nodes:
            ax.axis("off")
            continue

        g_mod = g_tier.subgraph(mod_nodes).copy()
        # Remove isolated nodes (degree 0 within the module tier subgraph)
        isolated = [n for n in g_mod.nodes() if g_mod.degree(n) == 0]
        if isolated:
            g_mod = g_mod.copy()
            g_mod.remove_nodes_from(isolated)
        if g_mod.number_of_nodes() == 0:
            ax.axis("off")
            continue

        pos = nx.spring_layout(g_mod, seed=7 + panel_idx, weight="weight", k=2.5, iterations=150)
        node_list = list(g_mod.nodes())
        sizes = degree_scaled_sizes(g_mod, lo=10.0, hi=100.0)
        fill_col = module_color(mod_id)
        borders = [cell_type_border_color(node_info.get(n, {}).get("cell_type")) for n in node_list]
        sz = [sizes.get(n, 30.0) for n in node_list]

        # Collect all cell types present for shared legend
        for n in node_list:
            ct_raw = node_info.get(n, {}).get("cell_type")
            if ct_raw and str(ct_raw).lower() not in {"nan", "none", ""}:
                key = str(ct_raw).lower().replace(" ", "_")
                all_ct_keys[key] = ct_display_name(ct_raw)

        hubs_in_mod = set(hub_genes or []) & set(node_list)

        nx.draw_networkx_edges(g_mod, pos, ax=ax,
                               edge_color=tier_color, alpha=0.25, width=0.5)
        # All nodes keep cell type border; hub nodes drawn larger on top
        non_hub_list = [n for n in node_list if n not in hubs_in_mod]
        hub_list = [n for n in node_list if n in hubs_in_mod]
        if non_hub_list:
            nx.draw_networkx_nodes(g_mod, pos, nodelist=non_hub_list, ax=ax,
                                   node_color=[fill_col] * len(non_hub_list),
                                   node_size=[sz[node_list.index(n)] for n in non_hub_list],
                                   alpha=0.75, linewidths=1.5,
                                   edgecolors=[borders[node_list.index(n)] for n in non_hub_list])
        if hub_list:
            nx.draw_networkx_nodes(g_mod, pos, nodelist=hub_list, ax=ax,
                                   node_color=[fill_col] * len(hub_list),
                                   node_size=[sz[node_list.index(n)] * 2.5 for n in hub_list],
                                   alpha=1.0, linewidths=2.5,
                                   edgecolors=[borders[node_list.index(n)] for n in hub_list])

        # Labels: hub genes always labelled with ◆ prefix, red bold; degree hubs smaller
        n_extra = max(5, min(15, len(node_list) // 5))
        degree_hubs = set(top_hub_genes(g_mod, n_extra))
        label_set = hubs_in_mod | degree_hubs
        texts = []
        for n in node_list:
            if n not in label_set:
                continue
            is_hub = n in hubs_in_mod
            info = node_info.get(n, {})
            csym = cancer_sym.get(str(info.get("cancer_role", "")), "")
            dsym = dir_sym.get(str(info.get("deg_direction", "")).lower(), "")
            prefix = "◆ " if is_hub else ""
            x, y = pos[n]
            texts.append(ax.text(x, y, f"{prefix}{n}{csym}{dsym}",
                                 fontsize=8.5 if is_hub else 7.0,
                                 color="#d62828" if is_hub else "#1d3557",
                                 fontweight="bold", ha="center", va="center"))
        if texts:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
                            expand_text=(1.2, 1.4), expand_points=(1.1, 1.2))

        # Dominant cell type for subtitle
        ct_counts: dict[str, int] = {}
        for n in node_list:
            ct = node_info.get(n, {}).get("cell_type")
            if ct and str(ct).lower() not in {"nan", "none", ""}:
                k = str(ct).lower().replace(" ", "_")
                ct_counts[k] = ct_counts.get(k, 0) + 1
        dominant_ct = ct_display_name(max(ct_counts, key=ct_counts.get)) if ct_counts else "—"

        ax.set_title(
            f"Module {mod_id}  ·  {mod_size} genes\ndominant: {dominant_ct}",
            fontsize=9, fontweight="bold", pad=4,
        )

        # Force square data range so spring layout appears circular
        xs = [pos[n][0] for n in pos]
        ys = [pos[n][1] for n in pos]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.08
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_aspect('equal')

        mod_legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=fill_col,
                   markeredgecolor="#444444", markeredgewidth=0.5,
                   markersize=10, label=f"Module {mod_id}")
        )
        ax.axis("off")

    # Shared bottom legend: Leiden modules + cell type borders + annotation symbols
    ct_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="none",
               markeredgecolor=CELL_TYPE_BORDER_COLORS.get(k, "#9CA3AF"),
               markeredgewidth=2.5, markersize=10, label=disp)
        for k, disp in sorted(all_ct_keys.items(), key=lambda x: x[1])[:10]
    ]
    hub_handle = mpatches.Patch(color="none", label="◆  Hub gene (larger, red label)")
    ann_handles = [
        hub_handle,
        mpatches.Patch(color="none", label="★  Oncogene"),
        mpatches.Patch(color="none", label="▲  TSG"),
        mpatches.Patch(color="none", label="↑  Up DEG"),
        mpatches.Patch(color="none", label="↓  Down DEG"),
        mpatches.Patch(color="none", label="−  Stable"),
    ]
    all_handles = mod_legend_handles + ct_handles + ann_handles
    if all_handles:
        ncols = min(8, max(len(mod_legend_handles), len(ct_handles) + len(ann_handles)))
        fig.legend(
            handles=all_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncols,
            frameon=True, framealpha=0.95, fontsize=8,
            borderpad=0.8, columnspacing=1.2,
            title="Leiden modules · Cell types · Annotation",
            title_fontsize=9,
        )

    display_name = pair.replace("__vs__", " vs ")
    fig.suptitle(
        f"{display_name}  |  Top 6 modules — {tier_display.get(tier, tier)}",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    logging.info("  %s (%d modules)", out_path.name, len(top6))


def build_annotated_pngs(
    pair: str,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    modules_map: dict[str, int],
    out_dir: Path,
    prefix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    display_name = pair.replace("__vs__", " vs ")
    p = f"{prefix}_" if prefix else ""

    # Each entry: (filename_stem, tier_filter, use_community_layout, min_mod_size, title_suffix)
    # 5 figures per network — each maps directly to a results section paragraph
    configs = [
        (
            f"{p}full_network.png",
            None,
            True, MIN_MOD_SIZE_FULL,
            "Full CSD network — all co-expression relationships",
        ),
        (
            f"{p}tier_CD.png",
            ["C", "D"],
            True, MIN_MOD_SIZE_FULL,
            "Shared network — conserved (C) and rewired (D) co-expression",
        ),
        (
            f"{p}tier_D.png",
            ["D"],
            True, MIN_MOD_SIZE_TIER,
            "Tier I — Rewired co-expression (D edges only)",
        ),
        (
            f"{p}tier_S_case.png",
            ["S_case"],
            True, MIN_MOD_SIZE_TIER,
            "Tier II — Tumour-gained co-expression (S+)",
        ),
        (
            f"{p}tier_S_ctrl.png",
            ["S_ctrl"],
            True, MIN_MOD_SIZE_TIER,
            "Tier III — Normal-lost co-expression (S−)",
        ),
    ]

    for fname, tier_filter, use_comm, min_mod, title_suffix in configs:
        g = _build_graph_from_edges(edges_df, tier_filter=tier_filter)
        if g.number_of_nodes() == 0:
            logging.warning("  %s: empty graph — skipping", fname)
            continue
        draw_annotated_network(
            g, out_dir / fname,
            f"{display_name} | {title_suffix}",
            nodes_df=nodes_df,
            modules_map=modules_map,
            use_community_layout=use_comm,
            min_mod_size=min_mod,
        )

    # Top 6 module panels — one 3×2 figure per tier; hub genes highlighted
    for tier, stem in [("D", "modules_D"), ("S_case", "modules_S_case"), ("S_ctrl", "modules_S_ctrl")]:
        tier_hubs_set = set(_tier_hubs(edges_df, nodes_df, tier)["gene"].tolist()
                            if "gene" in _tier_hubs(edges_df, nodes_df, tier).columns else [])
        draw_top6_module_panels(
            pair, tier, edges_df, nodes_df, modules_map,
            out_dir / f"{p}{stem}.png",
            hub_genes=tier_hubs_set,
        )


# ---------------------------------------------------------------------------
# Interactive HTML generation
# ---------------------------------------------------------------------------

def _html_node_obj(row: pd.Series, modules_map: dict[str, int]) -> dict:
    gene = str(row.get("gene", ""))
    wd = float(row["weighted_degree"]) if "weighted_degree" in row and pd.notna(row.get("weighted_degree")) else 0.0
    mod = int(modules_map.get(gene, row.get("module", 1) if pd.notna(row.get("module")) else 1))
    ct = str(row.get("cell_type", "")) if pd.notna(row.get("cell_type", "")) else ""
    cell_color = cell_type_border_color(ct if ct not in {"nan", "None", ""} else None)
    lnfc_val = row.get("lnFC", float("nan"))
    lnfc_str = f"{float(lnfc_val):+.3f}" if pd.notna(lnfc_val) else "n/a"
    cancer_role = str(row.get("cancer_role", "not_classified"))
    is_deg = bool(row.get("is_deg", False))
    deg_dir = str(row.get("deg_direction", ""))
    known_cancer = bool(row.get("known_cancer_gene", False))

    cancer_sym = {"oncogene": "★", "tsg": "▲"}.get(cancer_role.lower(), "")
    dir_sym = {"up": "↑", "down": "↓", "unchanged": "−"}.get(deg_dir.lower(), "")
    label = f"{gene}{cancer_sym}{dir_sym}" if (known_cancer or is_deg) else gene

    tooltip = (
        f"<b>{gene}</b><br>"
        f"Module: {mod}<br>"
        f"Cell type: {ct or 'Unknown'}<br>"
        f"Weighted degree: {wd:.4f}<br>"
        f"lnFC (DEG): {lnfc_str}<br>"
        f"DEG: {'Yes (' + deg_dir + ')' if is_deg else 'No'}<br>"
        f"Cancer: {cancer_role}"
    )
    return {
        "id": gene,
        "label": label,
        "original_label": gene,
        "group": f"C{mod}",
        "size": 10.0 + 30.0 * min(wd / 200.0, 1.0),
        "wd": wd,
        "cell_color": cell_color,
        "cell_type": ct or "Unknown",
        "borderWidth": 4,
        "borderWidthSelected": 6,
        "shape": "dot",
        "font": {"color": "#222222", "bold": {"color": "#1d3557"}},
        "title": tooltip,
    }


def _html_edge_obj(row: pd.Series) -> dict:
    lt = str(row.get("link_type_detail", "C"))
    w = float(row.get("weight", 1.0)) if pd.notna(row.get("weight")) else 1.0
    lt_visual = "S" if lt.startswith("S") else lt
    edge_label = {
        "C": "Conserved", "D": "Differential",
        "S": "Condition-specific",
    }.get(lt_visual, lt)
    return {
        "from": str(row["gene_a"]),
        "to": str(row["gene_b"]),
        "color": EDGE_COLORS.get(lt_visual, "#8d99ae"),
        "edge_type": lt_visual,   # used by edge type filter panel
        "value": float(row.get("wTO", w)),
        "ew": w,
        "width": min(0.5 + 6.0 * w / 2.0, 8.0),
        "title": f"Type: {edge_label}<br>Weight: {w:.4f}",
    }


def build_interactive_html(
    pair: str,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    modules_map: dict[str, int],
) -> str:
    if not TEMPLATE_HTML.exists():
        raise FileNotFoundError(f"template.html not found at {TEMPLATE_HTML}")

    template = TEMPLATE_HTML.read_text(encoding="utf-8")

    nodes_list = [_html_node_obj(row, modules_map) for _, row in nodes_df.iterrows()]
    edges_list = [_html_edge_obj(row) for _, row in edges_df.iterrows()]

    nodes_json = json.dumps(nodes_list, separators=(",", ":"))
    edges_json = json.dumps(edges_list, separators=(",", ":"))

    nodes_replacement = f"nodes = new vis.DataSet({nodes_json});"
    edges_replacement = f"edges = new vis.DataSet({edges_json});"
    html = re.sub(
        r"nodes\s*=\s*new vis\.DataSet\(\[.*?\]\);",
        lambda _: nodes_replacement,
        template, count=1, flags=re.DOTALL,
    )
    html = re.sub(
        r"edges\s*=\s*new vis\.DataSet\(\[.*?\]\);",
        lambda _: edges_replacement,
        html, count=1, flags=re.DOTALL,
    )
    display_name = pair.replace("__vs__", " vs ")
    html = re.sub(r"<h1></h1>", f"<h1>{display_name}</h1>", html, count=1)
    return html


# ---------------------------------------------------------------------------
# Per-pair orchestration
# ---------------------------------------------------------------------------

_STALE_PNG_NAMES = {
    "full_annotated.png", "full_network.png",
    "tier_CD.png", "tier_D.png", "tier_S_case.png", "tier_S_ctrl.png",
    "modules_case.png", "modules_ctrl.png", "modules_shared.png",
    "modules_D.png", "modules_S_case.png", "modules_S_ctrl.png",
    "plain_full_network.png", "plain_tier_CD.png", "plain_tier_D.png",
    "plain_tier_S_case.png", "plain_tier_S_ctrl.png",
}


def _cleanup_stale_files(pair_dir: Path, pair: str) -> None:
    """Remove stale files left by earlier script runs (old pair-prefixed CSVs, old unprefixed PNGs)."""
    for f in pair_dir.glob(f"{pair}_*.csv"):
        f.unlink()
        logging.info("  removed stale %s", f.name)
    fig_dir = pair_dir / "figures"
    if fig_dir.exists():
        for f in fig_dir.iterdir():
            if f.name in _STALE_PNG_NAMES:
                f.unlink()
                logging.info("  removed stale figures/%s", f.name)


def process_pair(pair: str) -> None:
    logging.info("Processing %s", pair)
    pair_dir = BASE / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    prefix = PAIR_SHORT.get(pair, pair)

    edges_df = load_edges(pair)
    modules_map = load_modules(pair)
    nodes_df = build_nodes_df(pair)

    nodes_df.to_csv(pair_dir / f"{prefix}_nodes.csv", index=False)
    logging.info("  %s_nodes.csv (%d rows)", prefix, len(nodes_df))

    modules_df = build_modules_df(nodes_df, edges_df)
    modules_df.to_csv(pair_dir / f"{prefix}_modules.csv", index=False)
    logging.info("  %s_modules.csv (%d modules)", prefix, len(modules_df))

    summary_df = build_summary_stats(pair, nodes_df, edges_df, modules_df)
    summary_df.to_csv(pair_dir / f"{prefix}_network_summary.csv", index=False)
    logging.info("  %s_network_summary.csv written", prefix)

    hubs_overall = build_hubs_overall(pair, nodes_df)
    hubs_overall.to_csv(pair_dir / f"{prefix}_hubs_overall.csv", index=False)
    logging.info("  %s_hubs_overall.csv (%d rows)", prefix, len(hubs_overall))

    for tier, stem in [("D", "hubs_D"), ("S_case", "hubs_S_case"), ("S_ctrl", "hubs_S_ctrl")]:
        hubs = _tier_hubs(edges_df, nodes_df, tier)
        fname = f"{prefix}_{stem}.csv"
        hubs.to_csv(pair_dir / fname, index=False)
        logging.info("  %s (%d hubs)", fname, len(hubs))

    # Annotated PNGs — all files prefixed with short name
    build_annotated_pngs(pair, edges_df, nodes_df, modules_map,
                         pair_dir / "figures", prefix=prefix)

    # Interactive HTML
    try:
        html = build_interactive_html(pair, edges_df, nodes_df, modules_map)
        (pair_dir / f"{prefix}_network_annotated.html").write_text(html, encoding="utf-8")
        logging.info("  %s_network_annotated.html written", prefix)
    except Exception as exc:
        logging.error("  HTML generation failed for %s: %s", pair, exc)

    # Clean up stale files from earlier runs
    _cleanup_stale_files(pair_dir, pair)


def main() -> None:
    configure_logging()
    logging.info("Script 47: Preparing visualization assets")

    missing_lfc = [p for p in PAIRS if not (LFC_DIR / f"{p}_tagged_with_lfc.csv").exists()]
    if missing_lfc:
        logging.error("Missing LFC files (run script 46 first): %s", missing_lfc)
        return

    done = 0
    for pair in PAIRS:
        try:
            process_pair(pair)
            done += 1
        except Exception as exc:
            logging.exception("Failed for %s: %s", pair, exc)

    logging.info("Completed %d/%d pairs", done, len(PAIRS))


if __name__ == "__main__":
    main()
