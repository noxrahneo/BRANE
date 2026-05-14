#!/usr/bin/env python3
"""Render condition-wise NetworkX visualizations from adjacency outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sparse NetworkX visualizations from adjacency matrices"
    )
    parser.add_argument(
        "--input-dir",
        default="results/11_coexpression_prep/single",
        help="Root directory containing per-condition adjacency outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/13_coexpression_viz/single",
        help="Output root for NetworkX figures and tables",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--power",
        type=int,
        default=None,
        help="Use adjacency at this beta; otherwise use selected power json",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned", "both"],
        default="signed",
        help="Adjacency network type file suffix",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=2500,
        help="Maximum strongest edges to keep for visualization (<=0 keeps all)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.10,
        help="Minimum adjacency weight for candidate edges",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for graph layout",
    )
    parser.add_argument(
        "--label-top-hubs",
        type=int,
        default=15,
        help="Number of highest-degree hubs to label in global plot",
    )
    parser.add_argument(
        "--hub-neighbors",
        type=int,
        default=40,
        help="Top neighbors to draw around the strongest hub",
    )
    parser.add_argument(
        "--ego-top-hubs",
        type=int,
        default=5,
        help="Number of top hubs for ego-network exports",
    )
    parser.add_argument(
        "--ego-radii",
        default="1,2",
        help="Comma-separated ego radii to export for each hub",
    )
    parser.add_argument(
        "--ego-max-nodes",
        type=int,
        default=150,
        help="Maximum nodes in exported ego subgraphs (largest weighted-degree nodes kept)",
    )
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada"],
        default="spring",
        help="Static PNG layout algorithm",
    )
    parser.add_argument(
        "--global-max-edges",
        type=int,
        default=800,
        help="Maximum strongest edges to draw in global PNG (<=0 keeps all)",
    )
    parser.add_argument(
        "--global-min-weight",
        type=float,
        default=0.20,
        help="Minimum adjacency weight to draw in global PNG (readability-focused)",
    )
    parser.add_argument(
        "--global-fig-width",
        type=float,
        default=18.0,
        help="Global PNG figure width in inches",
    )
    parser.add_argument(
        "--global-fig-height",
        type=float,
        default=14.0,
        help="Global PNG figure height in inches",
    )
    parser.add_argument(
        "--global-dpi",
        type=int,
        default=240,
        help="Global PNG DPI",
    )
    parser.add_argument(
        "--interactive-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export interactive HTML network (PyVis) when available",
    )
    parser.add_argument(
        "--interactive-max-edges",
        type=int,
        default=None,
        help="Maximum edges for interactive HTML graph (default: use --max-edges; <=0 keeps all)",
    )
    parser.add_argument(
        "--interactive-min-weight",
        type=float,
        default=None,
        help="Minimum adjacency weight for interactive HTML graph (default: use --min-weight)",
    )
    parser.add_argument(
        "--interactive-physics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable interactive physics (continuous movement) in HTML graph",
    )
    parser.add_argument(
        "--interactive-label-mode",
        choices=["hubs", "all", "none"],
        default="hubs",
        help="Node label density in interactive graph",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=1,
        help=(
            "Minimum node degree to keep in visualization graph "
            "(1 keeps current behavior, 2 hides degree<=1 genes)"
        ),
    )
    parser.add_argument(
        "--interactive-template",
        default="scripts/network_interactive_template.html",
        help="HTML template used for interactive outputs",
    )
    parser.add_argument(
        "--annotation-dir",
        default="results/04_annotation",
        help="Root directory containing per-condition annotation outputs",
    )
    parser.add_argument(
        "--interactive-celltype-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay cell-type annotation on interactive nodes when marker zscore files exist",
    )
    parser.add_argument(
        "--external-marker-files",
        nargs="*",
        default=[],
        help=(
            "Optional external marker DB files (csv/tsv/txt) with gene/cell_type "
            "columns; used as extra fallback for unknown genes"
        ),
    )
    parser.add_argument(
        "--external-marker-min-score",
        type=float,
        default=0.0,
        help="Minimum marker score/confidence to keep from external DB files",
    )
    parser.add_argument(
        "--external-marker-tissue-hint",
        default="",
        help="Optional tissue filter (case-insensitive contains match) for external DB rows",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def list_condition_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[Path]:
    dirs = list_condition_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No condition directories found in {root}")
    if requested.strip().lower() == "all":
        return dirs
    match = [d for d in dirs if d.name == requested]
    if not match:
        raise ValueError(f"Condition '{requested}' not found; available={[d.name for d in dirs]}")
    return match


def load_selected_power(cond_dir: Path) -> int | None:
    js = cond_dir / f"{cond_dir.name}_selected_power.json"
    if not js.exists():
        return None
    meta = json.loads(js.read_text(encoding="utf-8"))
    return int(meta.get("selected_power"))


def compute_layout(g: nx.Graph, seed: int, layout: str) -> dict[str, np.ndarray]:
    if layout == "kamada":
        return nx.kamada_kawai_layout(g, weight="weight")
    # Use fewer iterations for large graphs to avoid O(n²·iter) hangs
    iters = 30 if g.number_of_nodes() > 1000 else 300
    return nx.spring_layout(g, seed=int(seed), k=None, weight="weight", iterations=iters)


def resolve_adjacency_file(cond_dir: Path, power: int | None, network_type: str) -> Path:
    if power is None:
        p = load_selected_power(cond_dir)
    else:
        p = int(power)

    if p is not None:
        candidate = cond_dir / f"{cond_dir.name}_adjacency_beta{p}_{network_type}.npz"
        if candidate.exists():
            return candidate

    matches = sorted(cond_dir.glob(f"*_adjacency_beta*_{network_type}.npz"))
    if not matches:
        raise FileNotFoundError(f"No adjacency npz found in {cond_dir}")
    return matches[-1]


def top_edges_from_adjacency(adjacency: np.ndarray, min_weight: float, max_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(adjacency, dtype=np.float64)
    n = a.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    vals = a[tri_i, tri_j]

    keep = np.where(vals >= float(min_weight))[0]
    if keep.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    if int(max_edges) > 0 and keep.size > int(max_edges):
        sub_vals = vals[keep]
        top_idx = np.argpartition(sub_vals, -int(max_edges))[-int(max_edges):]
        keep = keep[top_idx]

    order = np.argsort(vals[keep])[::-1]
    keep = keep[order]
    return tri_i[keep], tri_j[keep], vals[keep]


def count_candidate_edges(adjacency: np.ndarray, min_weight: float) -> int:
    a = np.asarray(adjacency, dtype=np.float64)
    n = a.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    vals = a[tri_i, tri_j]
    return int(np.sum(vals >= float(min_weight)))


def parse_ego_radii(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        val = int(p)
        if val > 0:
            values.append(val)
    out = sorted(set(values))
    return out if out else [1, 2]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "item"


def build_graph_from_edges(i: np.ndarray, j: np.ndarray, w: np.ndarray, genes: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    for ii, jj, ww in zip(i, j, w):
        g.add_edge(str(genes[int(ii)]), str(genes[int(jj)]), weight=float(ww))
    return g


def filter_graph_by_min_degree(g: nx.Graph, min_degree: int) -> nx.Graph:
    # Use k-core so the final graph consistently satisfies the minimum degree.
    k = int(min_degree)
    if k <= 1 or g.number_of_nodes() == 0:
        return g
    return nx.k_core(g, k=k)


def community_palette() -> list[str]:
    return [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]


def cell_type_palette() -> list[str]:
    return [
        "#111827",
        "#7c3aed",
        "#dc2626",
        "#0f766e",
        "#d97706",
        "#be123c",
        "#1d4ed8",
        "#065f46",
        "#92400e",
        "#4c1d95",
        "#9f1239",
        "#164e63",
    ]


def normalize_gene_symbol(gene: str) -> str:
    return str(gene).strip().upper()


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def load_external_marker_mappings(
    marker_files: list[Path],
    tissue_hint: str = "",
    min_score: float = 0.0,
) -> dict[str, str]:
    if not marker_files:
        return {}

    gene_to_ct_score: dict[tuple[str, str], float] = {}
    tissue_hint_norm = tissue_hint.strip().lower()

    for marker_file in marker_files:
        if not marker_file.exists():
            continue

        sep = "\t" if marker_file.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            df = pd.read_csv(marker_file, sep=sep)
        except Exception:
            continue

        if df.empty:
            continue

        gene_col = pick_first_existing_column(
            df,
            ["gene", "gene_symbol", "symbol", "marker", "signatures", "markers"],
        )
        ct_col = pick_first_existing_column(
            df,
            ["cell_type", "celltype", "cell type", "CellType", "cell_name"],
        )
        if gene_col is None or ct_col is None:
            continue

        score_col = pick_first_existing_column(
            df,
            ["score", "specificity", "confidence", "evidence", "marker_score"],
        )
        tissue_col = pick_first_existing_column(
            df,
            ["tissue", "organ", "source_tissue", "tumor_type"],
        )

        work = df.copy()
        if tissue_hint_norm and tissue_col is not None:
            tissue_vals = work[tissue_col].fillna("").astype(str).str.lower()
            work = work.loc[tissue_vals.str.contains(tissue_hint_norm, na=False)].copy()

        if work.empty:
            continue

        if score_col is not None:
            scores = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
        else:
            scores = pd.Series(np.ones(len(work), dtype=float), index=work.index)

        for idx, row in work.iterrows():
            gene = row.get(gene_col)
            cell_type = row.get(ct_col)
            if pd.isna(gene) or pd.isna(cell_type):
                continue
            score = float(scores.loc[idx])
            if score < float(min_score):
                continue
            norm_gene = normalize_gene_symbol(str(gene))
            ct = str(cell_type).strip()
            if not norm_gene or not ct:
                continue
            key = (norm_gene, ct)
            gene_to_ct_score[key] = gene_to_ct_score.get(key, 0.0) + max(score, 0.0)

    best_by_gene: dict[str, tuple[str, float]] = {}
    for (gene, cell_type), score in gene_to_ct_score.items():
        prev = best_by_gene.get(gene)
        if prev is None or score > prev[1]:
            best_by_gene[gene] = (cell_type, score)

    return {gene: ct for gene, (ct, _) in best_by_gene.items()}


def load_cluster_marker_fallback(
    annotation_root: Path, condition: str
) -> dict[str, str]:
    markers_file = (
        annotation_root / condition / f"{condition}_cluster_markers_top.csv"
    )
    mapping_file = (
        annotation_root
        / condition
        / f"{condition}_cluster_to_celltype_mapping.csv"
    )
    if not markers_file.exists() or not mapping_file.exists():
        return {}

    try:
        markers = pd.read_csv(markers_file)
        mapping = pd.read_csv(mapping_file)
    except Exception:
        return {}

    if markers.empty or mapping.empty:
        return {}
    if "cluster" not in markers.columns or "names" not in markers.columns:
        return {}
    if "leiden" not in mapping.columns or "predicted_cell_type" not in mapping.columns:
        return {}

    cluster_to_type: dict[str, str] = {
        str(k): str(v)
        for k, v in zip(mapping["leiden"], mapping["predicted_cell_type"])
        if pd.notna(v)
    }
    if not cluster_to_type:
        return {}

    if "scores" in markers.columns:
        marker_scores = pd.to_numeric(markers["scores"], errors="coerce").fillna(1.0)
    else:
        marker_scores = pd.Series(np.ones(len(markers), dtype=float))

    gene_ct_score: dict[tuple[str, str], float] = {}
    for idx, row in markers.iterrows():
        cluster = str(row.get("cluster", ""))
        gene = row.get("names")
        if pd.isna(gene) or cluster not in cluster_to_type:
            continue
        cell_type = cluster_to_type[cluster]
        score = float(marker_scores.iloc[idx])
        norm_gene = normalize_gene_symbol(str(gene))
        if not norm_gene:
            continue
        key = (norm_gene, cell_type)
        gene_ct_score[key] = gene_ct_score.get(key, 0.0) + max(0.0, score)

    best_by_gene: dict[str, tuple[str, float]] = {}
    for (gene, cell_type), score in gene_ct_score.items():
        prev = best_by_gene.get(gene)
        if prev is None or score > prev[1]:
            best_by_gene[gene] = (cell_type, score)

    return {gene: cell_type for gene, (cell_type, _) in best_by_gene.items()}


def load_curated_immune_marker_fallback() -> dict[str, str]:
    candidate_files = [
        REPO_ROOT / "data/HumanBreast10X-main/Signatures/ImmuneMarkers2.txt",
        REPO_ROOT.parent / "HumanBreast10X-main/Signatures/ImmuneMarkers2.txt",
    ]
    marker_file = next((p for p in candidate_files if p.exists()), None)
    if marker_file is None:
        return {}

    try:
        markers = pd.read_csv(marker_file, sep="\t")
    except Exception:
        return {}

    if markers.empty or "CellType" not in markers.columns or "Signatures" not in markers.columns:
        return {}

    normalized_map = {
        "BCell": "BCell",
        "TCell": "TCell",
        "TCell2": "TCell",
        "NK": "NK",
        "DC": "DC",
        "Macro": "Myeloid",
        "Endo": "Endothelial",
        "Mega": "Megakaryocyte",
        "Fibro": "Fibroblast",
        "Fibro2": "Fibroblast",
    }

    out: dict[str, str] = {}
    for _, row in markers.iterrows():
        gene = row.get("Signatures")
        cell_type = row.get("CellType")
        if pd.isna(gene) or pd.isna(cell_type):
            continue
        norm_gene = normalize_gene_symbol(str(gene))
        if not norm_gene:
            continue
        canonical_ct = normalized_map.get(str(cell_type), str(cell_type))
        out.setdefault(norm_gene, canonical_ct)
    return out


def load_gene_to_cell_type(
    annotation_root: Path,
    condition: str,
    external_marker_files: list[Path] | None = None,
    external_marker_tissue_hint: str = "",
    external_marker_min_score: float = 0.0,
) -> dict[str, str]:
    zscore_file = annotation_root / condition / f"{condition}_marker_heatmap_zscore.csv"
    primary: dict[str, str] = {}
    if zscore_file.exists():
        try:
            zscores = pd.read_csv(zscore_file, index_col=0)
        except Exception:
            zscores = pd.DataFrame()

        if not zscores.empty:
            zscores = zscores.apply(pd.to_numeric, errors="coerce")
            if zscores.shape[1] > 0:
                top_cell_type = zscores.idxmax(axis=1)
                primary = {
                    normalize_gene_symbol(str(gene)): str(cell_type)
                    for gene, cell_type in top_cell_type.items()
                    if pd.notna(cell_type)
                }

    cluster_fallback = load_cluster_marker_fallback(annotation_root, condition)
    immune_fallback = load_curated_immune_marker_fallback()
    external_fallback = load_external_marker_mappings(
        marker_files=external_marker_files or [],
        tissue_hint=external_marker_tissue_hint,
        min_score=external_marker_min_score,
    )

    merged: dict[str, str] = {}
    merged.update(external_fallback)
    merged.update(immune_fallback)
    merged.update(cluster_fallback)
    merged.update(primary)
    return merged


def infer_cell_types_from_network_neighbors(
    g: nx.Graph,
    seed_gene_to_cell_type: dict[str, str],
    max_rounds: int = 2,
    min_best_fraction: float = 0.60,
    min_support_neighbors: int = 2,
) -> dict[str, str]:
    inferred = {
        normalize_gene_symbol(gene): str(cell_type)
        for gene, cell_type in seed_gene_to_cell_type.items()
        if str(cell_type) and str(cell_type) != "Unknown"
    }

    if not inferred or g.number_of_nodes() == 0:
        return inferred

    for _ in range(max_rounds):
        updates: dict[str, str] = {}
        for node in g.nodes():
            norm_node = normalize_gene_symbol(str(node))
            if norm_node in inferred:
                continue

            vote_weight: dict[str, float] = {}
            vote_count: dict[str, int] = {}
            for nbr in g.neighbors(node):
                norm_nbr = normalize_gene_symbol(str(nbr))
                nbr_type = inferred.get(norm_nbr)
                if not nbr_type:
                    continue
                edge_w = float(g[node][nbr].get("weight", 1.0))
                vote_weight[nbr_type] = vote_weight.get(nbr_type, 0.0) + max(edge_w, 0.0)
                vote_count[nbr_type] = vote_count.get(nbr_type, 0) + 1

            if not vote_weight:
                continue

            best_type, best_weight = max(vote_weight.items(), key=lambda x: x[1])
            total_weight = float(sum(vote_weight.values()))
            if total_weight <= 0:
                continue
            best_fraction = best_weight / total_weight
            if best_fraction < float(min_best_fraction):
                continue
            if vote_count.get(best_type, 0) < int(min_support_neighbors):
                continue

            updates[norm_node] = best_type

        if not updates:
            break
        inferred.update(updates)

    return inferred


def detect_communities(g: nx.Graph) -> tuple[dict[str, int], pd.DataFrame]:
    if g.number_of_nodes() == 0:
        return {}, pd.DataFrame(columns=["community", "n_nodes"])
    communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    communities = sorted(communities, key=len, reverse=True)
    node_to_comm: dict[str, int] = {}
    for idx, comm_nodes in enumerate(communities, start=1):
        for node in comm_nodes:
            node_to_comm[str(node)] = int(idx)
    comm_df = pd.DataFrame(
        {
            "community": np.arange(1, len(communities) + 1, dtype=int),
            "n_nodes": [len(c) for c in communities],
        }
    )
    return node_to_comm, comm_df


def export_edge_node_tables(g: nx.Graph, edges_file: Path, nodes_file: Path) -> pd.DataFrame:
    weighted_degree = dict(g.degree(weight="weight"))
    node_df = (
        pd.DataFrame(
            {
                "gene": list(weighted_degree.keys()),
                "weighted_degree": list(weighted_degree.values()),
                "degree": [int(g.degree(n)) for n in weighted_degree.keys()],
            }
        )
        .sort_values("weighted_degree", ascending=False)
        .reset_index(drop=True)
    )
    node_df.to_csv(nodes_file, index=False)

    rows: list[dict[str, float | str]] = []
    for u, v, d in g.edges(data=True):
        rows.append(
            {
                "gene_a": str(u),
                "gene_b": str(v),
                "weight": float(d.get("weight", 0.0)),
            }
        )
    edge_df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    edge_df.to_csv(edges_file, index=False)
    return node_df


def draw_global_graph(
    g: nx.Graph,
    out_file: Path,
    seed: int,
    label_top_hubs: int,
    title: str,
    layout: str,
    node_to_comm: dict[str, int],
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(max(6.0, float(fig_width)), max(4.0, float(fig_height))))
        ax.set_title(f"{title}\n(no edges after thresholding)")
        ax.axis("off")
        fig.savefig(out_file, dpi=int(max(72, dpi)), bbox_inches="tight")
        plt.close(fig)
        return pd.DataFrame(columns=["gene", "weighted_degree"])

    weighted_degree = dict(g.degree(weight="weight"))
    degree_df = (
        pd.DataFrame({"gene": list(weighted_degree.keys()), "weighted_degree": list(weighted_degree.values())})
        .sort_values("weighted_degree", ascending=False)
        .reset_index(drop=True)
    )

    pos = compute_layout(g, seed=seed, layout=layout)
    nodes = list(g.nodes())
    deg_vals = np.array([weighted_degree[n] for n in nodes], dtype=float)
    if deg_vals.size > 0 and np.max(deg_vals) > 0:
        node_sizes = 70 + 900 * (deg_vals / np.max(deg_vals))
    else:
        node_sizes = np.full(len(nodes), 120.0)

    edge_weights = np.array([d.get("weight", 0.1) for _, _, d in g.edges(data=True)], dtype=float)
    if edge_weights.size > 0 and np.max(edge_weights) > 0:
        edge_widths = 0.3 + 2.8 * (edge_weights / np.max(edge_weights))
    else:
        edge_widths = np.full(g.number_of_edges(), 0.6)

    fig, ax = plt.subplots(figsize=(max(6.0, float(fig_width)), max(4.0, float(fig_height))))
    nx.draw_networkx_edges(g, pos, ax=ax, width=edge_widths, alpha=0.25, edge_color="#7A8DAE")

    palette = community_palette()
    node_colors = [palette[(max(1, node_to_comm.get(str(n), 1)) - 1) % len(palette)] for n in nodes]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.90)

    top_labels = degree_df.head(int(max(0, label_top_hubs)))["gene"].tolist()
    labels = {n: n for n in top_labels}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)

    comm_counts = pd.Series([node_to_comm.get(str(n), 1) for n in nodes]).value_counts().sort_values(ascending=False)
    legend_items = comm_counts.head(6)
    for rank, (comm_id, n_nodes) in enumerate(legend_items.items(), start=1):
        ax.scatter([], [], c=palette[(comm_id - 1) % len(palette)], label=f"C{comm_id} (n={int(n_nodes)})")
    if len(legend_items) > 0:
        ax.legend(loc="upper left", frameon=True, title="Top communities")

    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_file, dpi=int(max(72, dpi)), bbox_inches="tight")
    plt.close(fig)
    return degree_df


def draw_hub_subgraph(g: nx.Graph, degree_df: pd.DataFrame, out_file: Path, seed: int, hub_neighbors: int, title: str) -> None:
    if g.number_of_nodes() == 0 or degree_df.empty:
        return

    hub = str(degree_df.iloc[0]["gene"])
    nbrs = list(g.neighbors(hub))
    if not nbrs:
        return

    weighted_nbrs: list[tuple[str, float]] = []
    for n in nbrs:
        weighted_nbrs.append((n, float(g[hub][n].get("weight", 0.0))))
    weighted_nbrs.sort(key=lambda x: x[1], reverse=True)
    keep = [hub] + [n for n, _ in weighted_nbrs[: int(max(1, hub_neighbors))]]

    sub = g.subgraph(keep).copy()
    pos = nx.spring_layout(sub, seed=int(seed), weight="weight")

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.35, edge_color="#4C78A8")
    node_sizes = [800 if n == hub else 260 for n in sub.nodes()]
    node_colors = ["#E45756" if n == hub else "#72B7B2" for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=8)

    ax.set_title(f"{title} | Hub-focused subgraph ({hub})")
    ax.axis("off")
    fig.savefig(out_file, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compute_topology_diagnostics(g: nx.Graph, condition: str) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if g.number_of_nodes() == 0:
        empty = {
            "condition": condition,
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "largest_component_nodes": 0,
            "largest_component_fraction": float("nan"),
            "global_clustering_coefficient": float("nan"),
            "avg_local_clustering": float("nan"),
            "avg_shortest_path_unweighted_giant": float("nan"),
            "avg_shortest_path_weighted_giant": float("nan"),
            "degree_mean": float("nan"),
            "degree_median": float("nan"),
            "degree_q90": float("nan"),
            "degree_max": float("nan"),
            "weighted_degree_mean": float("nan"),
            "weighted_degree_median": float("nan"),
            "weighted_degree_q90": float("nan"),
            "weighted_degree_max": float("nan"),
        }
        return empty, pd.DataFrame(columns=["component_id", "n_nodes", "n_edges", "density"])

    components = [set(c) for c in nx.connected_components(g)]
    components = sorted(components, key=len, reverse=True)
    giant_nodes = components[0]
    giant = g.subgraph(giant_nodes).copy()

    degree_vals = np.array([g.degree(n) for n in g.nodes()], dtype=float)
    wdegree_vals = np.array([g.degree(n, weight="weight") for n in g.nodes()], dtype=float)

    avg_sp_unweighted = float("nan")
    avg_sp_weighted = float("nan")
    if giant.number_of_nodes() > 1:
        try:
            avg_sp_unweighted = float(nx.average_shortest_path_length(giant, weight=None))
        except Exception:
            avg_sp_unweighted = float("nan")
        try:
            giant_dist = giant.copy()
            for u, v, d in giant_dist.edges(data=True):
                w = float(d.get("weight", 0.0))
                d["distance"] = 1.0 / max(w, 1e-12)
            avg_sp_weighted = float(nx.average_shortest_path_length(giant_dist, weight="distance"))
        except Exception:
            avg_sp_weighted = float("nan")

    diag = {
        "condition": condition,
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_components": int(len(components)),
        "largest_component_nodes": int(giant.number_of_nodes()),
        "largest_component_fraction": float(giant.number_of_nodes() / max(1, g.number_of_nodes())),
        "global_clustering_coefficient": float(nx.transitivity(g)),
        "avg_local_clustering": float(nx.average_clustering(g, weight="weight")),
        "avg_shortest_path_unweighted_giant": avg_sp_unweighted,
        "avg_shortest_path_weighted_giant": avg_sp_weighted,
        "degree_mean": float(np.mean(degree_vals)),
        "degree_median": float(np.median(degree_vals)),
        "degree_q90": float(np.quantile(degree_vals, 0.90)),
        "degree_max": float(np.max(degree_vals)),
        "weighted_degree_mean": float(np.mean(wdegree_vals)),
        "weighted_degree_median": float(np.median(wdegree_vals)),
        "weighted_degree_q90": float(np.quantile(wdegree_vals, 0.90)),
        "weighted_degree_max": float(np.max(wdegree_vals)),
    }

    component_rows: list[dict[str, float | int | str]] = []
    for idx, nodes in enumerate(components, start=1):
        sub = g.subgraph(nodes)
        n_nodes = int(sub.number_of_nodes())
        n_edges = int(sub.number_of_edges())
        density = float((2 * n_edges) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        component_rows.append(
            {
                "component_id": int(idx),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": density,
            }
        )
    component_df = pd.DataFrame(component_rows)
    return diag, component_df


def draw_ego_subgraph(sub: nx.Graph, out_file: Path, title: str, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    if sub.number_of_nodes() == 0:
        ax.set_title(f"{title}\n(empty)")
        ax.axis("off")
        fig.savefig(out_file, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(sub, seed=int(seed), weight="weight")
    wdeg = dict(sub.degree(weight="weight"))
    nodes = list(sub.nodes())
    sizes = np.array([wdeg.get(n, 0.0) for n in nodes], dtype=float)
    sizes = 80 + 600 * (sizes / max(1e-12, float(np.max(sizes)))) if sizes.size else np.array([])

    edges = list(sub.edges(data=True))
    w = np.array([float(d.get("weight", 0.0)) for _, _, d in edges], dtype=float)
    widths = 0.5 + 2.5 * (w / max(1e-12, float(np.max(w)))) if w.size else np.array([])

    nx.draw_networkx_edges(sub, pos, ax=ax, width=widths if len(widths) else 0.8, alpha=0.30, edge_color="#4C78A8")
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=sizes if len(sizes) else 140, node_color="#72B7B2", alpha=0.90)
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_file, dpi=190, bbox_inches="tight")
    plt.close(fig)


def export_top_hub_egonets(
    g: nx.Graph,
    condition: str,
    degree_df: pd.DataFrame,
    out_dir: Path,
    top_hubs: int,
    radii: list[int],
    max_nodes: int,
    seed: int,
) -> pd.DataFrame:
    if g.number_of_nodes() == 0 or degree_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "hub_gene",
                "radius",
                "n_nodes",
                "n_edges",
                "density",
                "nodes_file",
                "edges_file",
                "png_file",
            ]
        )

    ego_dir = out_dir / "ego_networks"
    ego_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    hub_genes = degree_df.head(int(max(0, top_hubs)))["gene"].astype(str).tolist()
    for hub in hub_genes:
        for radius in radii:
            ego = nx.ego_graph(g, hub, radius=int(radius))
            if ego.number_of_nodes() > int(max_nodes):
                wdeg = dict(ego.degree(weight="weight"))
                keep = [n for n, _ in sorted(wdeg.items(), key=lambda x: x[1], reverse=True)[: int(max_nodes)]]
                if hub not in keep:
                    keep = [hub] + keep[:-1]
                ego = ego.subgraph(keep).copy()

            gene_tag = safe_name(hub)
            base = f"{condition}_ego_{gene_tag}_r{int(radius)}"
            nodes_file = ego_dir / f"{base}_nodes.csv"
            edges_file = ego_dir / f"{base}_edges.csv"
            png_file = ego_dir / f"{base}.png"

            ndf = pd.DataFrame(
                {
                    "gene": [str(n) for n in ego.nodes()],
                    "degree": [int(ego.degree(n)) for n in ego.nodes()],
                    "weighted_degree": [float(ego.degree(n, weight="weight")) for n in ego.nodes()],
                }
            ).sort_values("weighted_degree", ascending=False)
            ndf.to_csv(nodes_file, index=False)

            edf = pd.DataFrame(
                [
                    {
                        "gene_a": str(u),
                        "gene_b": str(v),
                        "weight": float(d.get("weight", 0.0)),
                    }
                    for u, v, d in ego.edges(data=True)
                ]
            )
            if not edf.empty:
                edf = edf.sort_values("weight", ascending=False)
            edf.to_csv(edges_file, index=False)

            draw_ego_subgraph(
                ego,
                out_file=png_file,
                title=f"{condition} ego-network | hub={hub} | radius={radius}",
                seed=int(seed),
            )

            n_nodes = int(ego.number_of_nodes())
            n_edges = int(ego.number_of_edges())
            density = float((2 * n_edges) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
            rows.append(
                {
                    "condition": condition,
                    "hub_gene": hub,
                    "radius": int(radius),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "density": density,
                    "nodes_file": str(nodes_file),
                    "edges_file": str(edges_file),
                    "png_file": str(png_file),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(ego_dir / f"{condition}_ego_networks_summary.csv", index=False)
    return summary_df


def draw_interactive_html(
    g: nx.Graph,
    out_file: Path,
    title: str,
    physics: bool,
    node_to_comm: dict[str, int],
    label_mode: str,
    gene_to_cell_type: dict[str, str] | None = None,
    template_file: Path | None = None,
) -> bool:
    def tooltip_list(lines: list[str]) -> str:
        clean = [str(x) for x in lines if str(x).strip()]
        if not clean:
            return ""

        # vis-network tooltips sanitize HTML strings; use plain text with line breaks.
        return "\n".join(clean)

    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)
    net.barnes_hut()

    weighted_degree = dict(g.degree(weight="weight"))
    max_deg = max(weighted_degree.values()) if weighted_degree else 1.0
    palette = community_palette()
    ct_palette = cell_type_palette()
    seed_gene_to_cell_type = {
        normalize_gene_symbol(gene): str(cell_type)
        for gene, cell_type in (gene_to_cell_type or {}).items()
        if pd.notna(cell_type)
    }
    gene_to_cell_type = infer_cell_types_from_network_neighbors(
        g,
        seed_gene_to_cell_type=seed_gene_to_cell_type,
    )
    unknown_cell_color = "#9ca3af"
    cell_type_to_color: dict[str, str] = {"Unknown": unknown_cell_color}
    next_cell_color_idx = 0
    top_hubs = set(
        pd.DataFrame(
            {"gene": list(weighted_degree.keys()), "weighted_degree": list(weighted_degree.values())}
        )
        .sort_values("weighted_degree", ascending=False)
        .head(30)["gene"]
        .tolist()
    )

    for node in g.nodes():
        deg = float(weighted_degree.get(node, 0.0))
        size = 10.0 + 35.0 * (deg / max(1e-12, max_deg))
        comm_id = int(node_to_comm.get(str(node), 1))
        module_color = palette[(comm_id - 1) % len(palette)]
        cell_type = str(
            gene_to_cell_type.get(normalize_gene_symbol(str(node)), "Unknown")
        )
        if cell_type not in cell_type_to_color:
            cell_type_to_color[cell_type] = ct_palette[next_cell_color_idx % len(ct_palette)]
            next_cell_color_idx += 1
        cell_color = cell_type_to_color[cell_type]
        if label_mode == "all":
            label = str(node)
        elif label_mode == "none":
            label = ""
        else:
            label = str(node) if node in top_hubs else ""
        node_meta = g.nodes[node] if node in g.nodes else {}
        csd_lines: list[str] = []
        if "kC" in node_meta:
            csd_lines.append(f"kC={int(node_meta.get('kC', 0))}")
        if "kS" in node_meta:
            csd_lines.append(f"kS={int(node_meta.get('kS', 0))}")
        if "kD" in node_meta:
            csd_lines.append(f"kD={int(node_meta.get('kD', 0))}")
        if "homogeneity" in node_meta:
            try:
                csd_lines.append(
                    f"H={float(node_meta.get('homogeneity', float('nan'))):.3f}"
                )
            except Exception:
                pass
        if "deg_direction" in node_meta:
            csd_lines.append(f"deg_direction={str(node_meta.get('deg_direction'))}")
        if "lnFC" in node_meta and pd.notna(node_meta.get("lnFC")):
            try:
                csd_lines.append(f"lnFC={float(node_meta.get('lnFC')):.3f}")
            except Exception:
                pass
        if "fdr" in node_meta and pd.notna(node_meta.get("fdr")):
            try:
                csd_lines.append(f"fdr={float(node_meta.get('fdr')):.3g}")
            except Exception:
                pass

        title_parts = [
            f"{node}",
            f"community=C{comm_id}",
            f"cell_type={cell_type}",
            f"weighted_degree={deg:.4f}",
        ]
        title_parts.extend(csd_lines)

        net.add_node(
            str(node),
            label=label,
            title=tooltip_list(title_parts),
            size=size,
            wd=deg,
            original_label=str(node),
            cell_type=cell_type,
            cell_color=cell_color,
            color={
                "background": module_color,
                "border": cell_color,
                "highlight": {"background": module_color, "border": cell_color},
            },
            borderWidth=4,
            borderWidthSelected=6,
            group=f"C{comm_id}",
        )

    max_w = max((float(d.get("weight", 0.0)) for _, _, d in g.edges(data=True)), default=1.0)
    edge_type_colors = {
        "C": "#2a9d8f",
        "S": "#f4a261",
        "D": "#e76f51",
    }
    for u, v, d in g.edges(data=True):
        w = float(d.get("weight", 0.0))
        width = 0.4 + 6.0 * (w / max(1e-12, max_w))
        link_type = str(d.get("link_type", ""))
        edge_color = edge_type_colors.get(link_type, "#8AA5CF")
        title_parts = [f"weight={w:.4f}"]
        if link_type in edge_type_colors:
            title_parts.insert(0, f"type={link_type}")
        if "selected_value" in d and pd.notna(d.get("selected_value")):
            try:
                title_parts.append(f"selected_value={float(d.get('selected_value')):.4f}")
            except Exception:
                pass
        if "C" in d and pd.notna(d.get("C")):
            try:
                title_parts.append(f"C={float(d.get('C')):.4f}")
            except Exception:
                pass
        if "S" in d and pd.notna(d.get("S")):
            try:
                title_parts.append(f"S={float(d.get('S')):.4f}")
            except Exception:
                pass
        if "D" in d and pd.notna(d.get("D")):
            try:
                title_parts.append(f"D={float(d.get('D')):.4f}")
            except Exception:
                pass
        if "delta_r" in d and pd.notna(d.get("delta_r")):
            try:
                title_parts.append(f"delta_r={float(d.get('delta_r')):.4f}")
            except Exception:
                pass
        net.add_edge(
            str(u),
            str(v),
            value=max(0.1, 8.0 * w),
            width=width,
            ew=w,
            title=tooltip_list(title_parts),
            color=edge_color,
        )

    net.toggle_physics(bool(physics))
    net.show_buttons(filter_=["physics"])

    net.save_graph(str(out_file))
    generated_html = out_file.read_text(encoding="utf-8")

    nodes_pattern = re.compile(r"nodes\s*=\s*new vis\.DataSet\(\[.*?\]\);", re.DOTALL)
    edges_pattern = re.compile(r"edges\s*=\s*new vis\.DataSet\(\[.*?\]\);", re.DOTALL)
    graph_title_pattern = re.compile(
        r"<h2\s+class=\"graph-title\">.*?</h2>", re.DOTALL
    )

    nodes_match = nodes_pattern.search(generated_html)
    edges_match = edges_pattern.search(generated_html)
    if not nodes_match or not edges_match:
        out_file.write_text(generated_html, encoding="utf-8")
        return True

    template_html = ""
    if template_file is not None and template_file.exists():
        template_html = template_file.read_text(encoding="utf-8")
    elif out_file.exists():
        template_html = out_file.read_text(encoding="utf-8")

    if not template_html:
        out_file.write_text(generated_html, encoding="utf-8")
        return True

    updated_html, nodes_replaced = nodes_pattern.subn(
        lambda _: nodes_match.group(0), template_html, count=1
    )
    updated_html, edges_replaced = edges_pattern.subn(
        lambda _: edges_match.group(0), updated_html, count=1
    )

    if "<title>" in updated_html:
        updated_html = updated_html.replace("<title></title>", f"<title>{title}</title>")

    condition = title.split(":", 1)[0].strip() if ":" in title else title
    graph_title = f"{condition} network (interactive)"
    updated_html = graph_title_pattern.sub(
        f'<h2 class="graph-title">{graph_title}</h2>',
        updated_html,
        count=1,
    )

    if nodes_replaced and edges_replaced:
        out_file.write_text(updated_html, encoding="utf-8")
        return True

    out_file.write_text(generated_html, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    interactive_template = resolve_base(args.interactive_template)
    annotation_root = resolve_base(args.annotation_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[WarehouseRecord] = []
    network_types = ["signed", "unsigned"] if args.network_type == "both" else [args.network_type]

    for network_type in network_types:
        type_in_root = in_root / network_type if args.network_type == "both" else in_root
        cond_dirs = resolve_conditions(type_in_root, args.condition)
        type_out_root = out_root / network_type if args.network_type == "both" else out_root
        type_out_root.mkdir(parents=True, exist_ok=True)

        for cond_dir in cond_dirs:
            condition = cond_dir.name
            adj_file = resolve_adjacency_file(cond_dir, args.power, network_type)
            payload = np.load(adj_file, allow_pickle=True)
            adjacency = np.asarray(payload["adjacency"], dtype=np.float64)
            genes = payload["genes"].astype(str)
            i, j, w = top_edges_from_adjacency(
                adjacency=adjacency,
                min_weight=args.min_weight,
                max_edges=args.max_edges,
            )
            candidate_edges = count_candidate_edges(
                adjacency=adjacency,
                min_weight=float(args.min_weight),
            )
            g = build_graph_from_edges(i=i, j=j, w=w, genes=genes)
            g = filter_graph_by_min_degree(g, min_degree=int(args.min_degree))
            node_to_comm, comm_df = detect_communities(g)

            cond_out = type_out_root / condition
            cond_out.mkdir(parents=True, exist_ok=True)

            edges_csv = cond_out / f"{condition}_network_edges.csv"
            nodes_csv = cond_out / f"{condition}_network_nodes.csv"
            degree_df = export_edge_node_tables(g, edges_file=edges_csv, nodes_file=nodes_csv)

            diagnostics_csv = cond_out / f"{condition}_network_edge_diagnostics.csv"
            pd.DataFrame(
                [
                    {
                        "condition": condition,
                        "n_genes_input": int(len(genes)),
                        "min_weight": float(args.min_weight),
                        "max_edges_requested": int(args.max_edges),
                        "min_degree": int(args.min_degree),
                        "candidate_edges_above_min_weight": int(candidate_edges),
                        "edges_kept_after_clipping": int(g.number_of_edges()),
                        "clipped_to_max_edges": bool(
                            int(args.max_edges) > 0
                            and int(candidate_edges) > int(args.max_edges)
                        ),
                    }
                ]
            ).to_csv(diagnostics_csv, index=False)

            global_i, global_j, global_w = top_edges_from_adjacency(
                adjacency=adjacency,
                min_weight=float(args.global_min_weight),
                max_edges=int(args.global_max_edges),
            )
            g_global = build_graph_from_edges(i=global_i, j=global_j, w=global_w, genes=genes)
            g_global = filter_graph_by_min_degree(g_global, min_degree=int(args.min_degree))
            global_node_to_comm, _ = detect_communities(g_global)

            global_png = cond_out / f"{condition}_network_global.png"
            _ = draw_global_graph(
                g_global,
                out_file=global_png,
                seed=args.seed,
                label_top_hubs=args.label_top_hubs,
                title=f"{condition}: sparse co-expression network (NetworkX)",
                layout=args.layout,
                node_to_comm=global_node_to_comm,
                fig_width=args.global_fig_width,
                fig_height=args.global_fig_height,
                dpi=args.global_dpi,
            )

            degree_csv = cond_out / f"{condition}_network_weighted_degree.csv"
            degree_df.to_csv(degree_csv, index=False)

            topo_diag, component_df = compute_topology_diagnostics(g=g, condition=condition)
            topology_csv = cond_out / f"{condition}_network_topology_metrics.csv"
            component_csv = cond_out / f"{condition}_network_component_stats.csv"
            pd.DataFrame([topo_diag]).to_csv(topology_csv, index=False)
            component_df.to_csv(component_csv, index=False)

            hub_png = cond_out / f"{condition}_network_hub_subgraph.png"
            draw_hub_subgraph(
                g,
                degree_df=degree_df,
                out_file=hub_png,
                seed=args.seed,
                hub_neighbors=args.hub_neighbors,
                title=condition,
            )

            comm_csv = cond_out / f"{condition}_network_community_sizes.csv"
            comm_df.to_csv(comm_csv, index=False)

            ego_summary = export_top_hub_egonets(
                g=g,
                condition=condition,
                degree_df=degree_df,
                out_dir=cond_out,
                top_hubs=int(args.ego_top_hubs),
                radii=parse_ego_radii(args.ego_radii),
                max_nodes=int(args.ego_max_nodes),
                seed=int(args.seed),
            )
            ego_summary_file = cond_out / "ego_networks" / f"{condition}_ego_networks_summary.csv"

            gexf_file = cond_out / f"{condition}_network_sparse.gexf"
            graphml_file = cond_out / f"{condition}_network_sparse.graphml"
            nx.write_gexf(g, gexf_file)
            nx.write_graphml(g, graphml_file)

            interactive_html_file = cond_out / f"{condition}_network_interactive.html"
            interactive_ok = False
            gene_to_cell_type = {}
            if args.interactive_html:
                if args.interactive_celltype_overlay:
                    external_files = [
                        resolve_base(p) for p in args.external_marker_files
                    ]
                    gene_to_cell_type = load_gene_to_cell_type(
                        annotation_root=annotation_root,
                        condition=condition,
                        external_marker_files=external_files,
                        external_marker_tissue_hint=args.external_marker_tissue_hint,
                        external_marker_min_score=float(args.external_marker_min_score),
                    )
                if args.interactive_max_edges is None and args.interactive_min_weight is None:
                    g_inter = g
                    inter_node_to_comm = node_to_comm
                else:
                    inter_max_edges = int(
                        args.interactive_max_edges if args.interactive_max_edges is not None else args.max_edges
                    )
                    inter_min_weight = float(
                        args.interactive_min_weight if args.interactive_min_weight is not None else args.min_weight
                    )
                    ii, jj, ww = top_edges_from_adjacency(
                        adjacency=adjacency,
                        min_weight=inter_min_weight,
                        max_edges=inter_max_edges,
                    )
                    g_inter = build_graph_from_edges(i=ii, j=jj, w=ww, genes=genes)
                    g_inter = filter_graph_by_min_degree(
                        g_inter, min_degree=int(args.min_degree)
                    )
                    inter_node_to_comm, _ = detect_communities(g_inter)
                interactive_ok = draw_interactive_html(
                    g_inter,
                    out_file=interactive_html_file,
                    title=f"{condition}: interactive co-expression network",
                    physics=args.interactive_physics,
                    node_to_comm=inter_node_to_comm,
                    label_mode=args.interactive_label_mode,
                    gene_to_cell_type=gene_to_cell_type,
                    template_file=interactive_template,
                )

            summary = {
                "condition": condition,
                "adjacency_file": str(adj_file),
                "n_nodes": int(g.number_of_nodes()),
                "n_edges": int(g.number_of_edges()),
                "min_weight": float(args.min_weight),
                "max_edges_requested": int(args.max_edges),
                "min_degree": int(args.min_degree),
                "global_min_weight": float(args.global_min_weight),
                "global_max_edges": int(args.global_max_edges),
                "network_type": network_type,
                "layout": args.layout,
                "n_genes_input": int(len(genes)),
                "candidate_edges_above_min_weight": int(candidate_edges),
                "clipped_to_max_edges": bool(
                    int(args.max_edges) > 0
                    and int(candidate_edges) > int(args.max_edges)
                ),
                "edges_table": str(edges_csv),
                "nodes_table": str(nodes_csv),
                "edge_diagnostics_table": str(diagnostics_csv),
                "gexf_file": str(gexf_file),
                "graphml_file": str(graphml_file),
                "interactive_html_requested": bool(args.interactive_html),
                "interactive_html_generated": bool(interactive_ok),
                "interactive_html_file": str(interactive_html_file) if interactive_ok else "",
                "interactive_celltype_overlay": bool(args.interactive_celltype_overlay),
                "interactive_celltype_annotated_genes": int(len(gene_to_cell_type)) if args.interactive_html else 0,
                "community_sizes_file": str(comm_csv),
                "topology_metrics_file": str(topology_csv),
                "component_stats_file": str(component_csv),
                "ego_network_summary_file": str(ego_summary_file) if not ego_summary.empty else "",
                "n_ego_networks": int(ego_summary.shape[0]),
            }
            summary_file = cond_out / f"{condition}_network_viz_summary.json"
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            records.append(
                WarehouseRecord(
                    input_file=str(adj_file),
                    output_file=str(summary_file),
                    script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                    date_utc=utc_now_iso(),
                    params_hash=params_hash(vars(args)),
                    condition=f"{condition}::{network_type}",
                    stage="08d_networkx_visualization",
                )
            )

            inter_msg = "interactive=ok" if interactive_ok else "interactive=not-available"
            print(
                f"[{condition}::{network_type}] nodes={g.number_of_nodes()}, "
                f"edges={g.number_of_edges()}, {inter_msg} -> {cond_out}"
            )

    append_warehouse(out_root, records)
    print(f"Done. Network visualization outputs: {out_root}")


if __name__ == "__main__":
    main()
