#!/usr/bin/env python3
"""Visualize differential C/S/D networks from script 37 outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False


REPO_ROOT = Path(__file__).resolve().parents[1]

EDGE_COLORS = {
    "C": "#2a9d8f",
    "S": "#f4a261",
    "D": "#e76f51",
}

NODE_COLORS = {
    "up": "#d62828",
    "down": "#1d4ed8",
    "unchanged": "#6b7280",
    "unknown": "#264653",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render static and interactive visualizations for differential C/S/D networks"
    )
    parser.add_argument(
        "--input-dir",
        default="results/07_network/zzz_12_differential_scalefree",
        help="Root directory with pair folders from script 37",
    )
    parser.add_argument(
        "--output-dir",
        default="results/13_coexpression_viz/zzz_differential_scalefree",
        help="Output root for visualizations",
    )
    parser.add_argument(
        "--pair",
        default="all",
        help="Pair folder name or 'all'",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=1200,
        help="Maximum number of edges to visualize (by selected_value or weight)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Minimum edge weight to keep",
    )
    parser.add_argument(
        "--min-node-degree",
        type=int,
        default=1,
        help="Drop nodes below this degree after edge filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Layout seed",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=16.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=12.0,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--interactive-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export interactive HTML with PyVis when available",
    )
    parser.add_argument(
        "--interactive-template",
        default="scripts/network_interactive_template.html",
        help="HTML template used for interactive outputs",
    )
    parser.add_argument(
        "--interactive-physics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable physics in interactive output",
    )
    parser.add_argument(
        "--interactive-label-mode",
        choices=["hubs", "all", "none"],
        default="hubs",
        help="Label density in interactive output",
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


def list_pair_dirs(root: Path, requested: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    dirs = sorted([p for p in root.iterdir() if p.is_dir() and "__vs__" in p.name])
    if not dirs:
        raise ValueError(f"No pair directories found under {root}")

    if requested.strip().lower() == "all":
        return dirs

    match = [p for p in dirs if p.name == requested]
    if not match:
        raise ValueError(
            f"Pair '{requested}' not found. Available: {[d.name for d in dirs]}"
        )
    return match


def read_edges(pair_dir: Path) -> pd.DataFrame:
    edges_file = pair_dir / f"{pair_dir.name}_differential_edges_scalefree.csv"
    if not edges_file.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_file}")
    df = pd.read_csv(edges_file)
    if df.empty:
        return df

    for col in ["weight", "selected_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["link_type"] = df.get("link_type", "C").astype(str)
    return df


def read_node_stats(pair_dir: Path) -> pd.DataFrame:
    nodes_file = pair_dir / f"{pair_dir.name}_node_homogeneity_scalefree.csv"
    if not nodes_file.exists():
        return pd.DataFrame(columns=["gene", "weighted_degree", "homogeneity"])
    return pd.read_csv(nodes_file)


def filter_edges(edges: pd.DataFrame, min_weight: float, max_edges: int) -> pd.DataFrame:
    if edges.empty:
        return edges
    out = edges.copy()
    if "weight" in out.columns:
        out = out[out["weight"] >= float(min_weight)].copy()

    rank_col = "selected_value" if "selected_value" in out.columns else "weight"
    out = out.sort_values(rank_col, ascending=False)
    if int(max_edges) > 0:
        out = out.head(int(max_edges)).copy()
    return out.reset_index(drop=True)


def build_graph(edges: pd.DataFrame, min_node_degree: int) -> nx.Graph:
    g = nx.Graph()
    if edges.empty:
        return g

    for row in edges.to_dict(orient="records"):
        g.add_edge(
            str(row["gene_a"]),
            str(row["gene_b"]),
            weight=float(row.get("weight", 1.0)),
            link_type=str(row.get("link_type", "C")),
            selected_value=float(row.get("selected_value", np.nan)),
            C=float(row.get("C", np.nan)),
            S=float(row.get("S", np.nan)),
            D=float(row.get("D", np.nan)),
            delta_r=float(row.get("delta_r", np.nan)),
            rho_case=float(row.get("rho_case", np.nan)),
            rho_control=float(row.get("rho_control", np.nan)),
        )

    if int(min_node_degree) > 1 and g.number_of_nodes() > 0:
        drop_nodes = [n for n, d in g.degree() if int(d) < int(min_node_degree)]
        g.remove_nodes_from(drop_nodes)
    return g


def node_sizes_from_degree(g: nx.Graph) -> dict[str, float]:
    if g.number_of_nodes() == 0:
        return {}
    wd = dict(g.degree(weight="weight"))
    vals = np.array(list(wd.values()), dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return {k: 110.0 for k in wd}
    return {
        k: float(80.0 + 380.0 * ((v - vmin) / (vmax - vmin)))
        for k, v in wd.items()
    }


def load_single_viz_module():
    script_path = REPO_ROOT / "scripts" / "26_networkx_visualization.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("single_viz26", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                "link_type": str(d.get("link_type", "C")),
                "selected_value": float(d.get("selected_value", np.nan)),
            }
        )
    edge_df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    edge_df.to_csv(edges_file, index=False)
    return node_df


def compute_topology_diagnostics(g: nx.Graph, pair_name: str) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if g.number_of_nodes() == 0:
        empty = {
            "pair": pair_name,
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "largest_component_nodes": 0,
            "largest_component_fraction": float("nan"),
            "global_clustering_coefficient": float("nan"),
            "avg_local_clustering": float("nan"),
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

    degree_vals = np.array([g.degree(n) for n in g.nodes()], dtype=float)
    wdegree_vals = np.array([g.degree(n, weight="weight") for n in g.nodes()], dtype=float)

    diag = {
        "pair": pair_name,
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_components": int(len(components)),
        "largest_component_nodes": int(len(giant_nodes)),
        "largest_component_fraction": float(len(giant_nodes) / max(1, g.number_of_nodes())),
        "global_clustering_coefficient": float(nx.transitivity(g)),
        "avg_local_clustering": float(nx.average_clustering(g, weight="weight")),
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
    return diag, pd.DataFrame(component_rows)


def compute_csd_feature_summary(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    pair_name: str,
) -> pd.DataFrame:
    edge_summary = {
        "pair": pair_name,
        "scope": "edges",
        "n_total": int(edges.shape[0]),
        "n_C": int((edges.get("link_type", pd.Series(dtype=str)) == "C").sum()) if not edges.empty else 0,
        "n_S": int((edges.get("link_type", pd.Series(dtype=str)) == "S").sum()) if not edges.empty else 0,
        "n_D": int((edges.get("link_type", pd.Series(dtype=str)) == "D").sum()) if not edges.empty else 0,
    }
    edge_total = max(edge_summary["n_total"], 1)
    edge_summary["frac_C"] = float(edge_summary["n_C"] / edge_total)
    edge_summary["frac_S"] = float(edge_summary["n_S"] / edge_total)
    edge_summary["frac_D"] = float(edge_summary["n_D"] / edge_total)

    node_summary = {
        "pair": pair_name,
        "scope": "nodes",
        "n_total": int(nodes.shape[0]),
        "n_up": int((nodes.get("deg_direction", pd.Series(dtype=str)) == "up").sum()) if not nodes.empty else 0,
        "n_down": int((nodes.get("deg_direction", pd.Series(dtype=str)) == "down").sum()) if not nodes.empty else 0,
        "n_unchanged": int((nodes.get("deg_direction", pd.Series(dtype=str)) == "unchanged").sum()) if not nodes.empty else 0,
    }
    node_total = max(node_summary["n_total"], 1)
    node_summary["frac_up"] = float(node_summary["n_up"] / node_total)
    node_summary["frac_down"] = float(node_summary["n_down"] / node_total)
    node_summary["frac_unchanged"] = float(node_summary["n_unchanged"] / node_total)

    return pd.DataFrame([edge_summary, node_summary])


def draw_hub_subgraph(
    g: nx.Graph,
    degree_df: pd.DataFrame,
    out_file: Path,
    seed: int,
    hub_neighbors: int,
    title: str,
) -> None:
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


def draw_png(g: nx.Graph, out_png: Path, pair_name: str, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(
        figsize=(float(args.fig_width), float(args.fig_height)),
        constrained_layout=True,
    )

    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges passed filters", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(g, seed=int(args.seed), weight="weight", iterations=200)
    node_sizes = node_sizes_from_degree(g)

    edge_groups = {"C": [], "S": [], "D": [], "other": []}
    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "other"))
        if t not in edge_groups:
            t = "other"
        edge_groups[t].append((u, v))

    for t, edges in edge_groups.items():
        if not edges:
            continue
        color = EDGE_COLORS.get(t, "#8d99ae")
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edges,
            width=0.8,
            alpha=0.55,
            edge_color=color,
            ax=ax,
        )

    node_colors = []
    for n in g.nodes():
        d = g.nodes[n]
        direction = str(d.get("deg_direction", "unknown"))
        node_colors.append(NODE_COLORS.get(direction, NODE_COLORS["unknown"]))

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=[node_sizes.get(n, 100.0) for n in g.nodes()],
        node_color=node_colors,
        alpha=0.88,
        linewidths=0.25,
        edgecolors="#f1faee",
        ax=ax,
    )

    top_nodes = sorted(g.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:20]
    labels = {n: n for n, _ in top_nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="#1d3557", ax=ax)

    legend_lines = []
    legend_labels = []
    for t in ["C", "S", "D"]:
        line = plt.Line2D([0], [0], color=EDGE_COLORS[t], lw=2)
        legend_lines.append(line)
        legend_labels.append(f"{t} edges")
    for k in ["up", "down", "unchanged"]:
        marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NODE_COLORS[k],
            markeredgecolor="#f1faee",
            markersize=8,
        )
        legend_lines.append(marker)
        legend_labels.append(f"{k} genes")
    ax.legend(legend_lines, legend_labels, loc="upper right", frameon=False)

    ax.set_title(
        f"{pair_name} differential C/S/D network\n"
        f"nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )
    ax.set_axis_off()
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def draw_html(g: nx.Graph, out_html: Path, pair_name: str, node_stats: pd.DataFrame) -> None:
    if not HAS_PYVIS:
        return

    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#1f2937")
    net.barnes_hut(gravity=-12000, central_gravity=0.2, spring_length=170, spring_strength=0.02)

    stats = node_stats.copy()
    if not stats.empty and "gene" in stats.columns:
        stats["gene"] = stats["gene"].astype(str)
        stats = stats.set_index("gene", drop=False)

    for n, deg_w in g.degree(weight="weight"):
        title = f"gene: {n}<br>weighted_degree: {deg_w:.3f}"
        node_color = NODE_COLORS["unknown"]
        if not stats.empty and n in stats.index:
            row = stats.loc[n]
            if "homogeneity" in row:
                title += f"<br>homogeneity: {float(row['homogeneity']):.3f}"
            if "deg_direction" in row:
                direction = str(row["deg_direction"])
                title += f"<br>deg_direction: {direction}"
                node_color = NODE_COLORS.get(direction, NODE_COLORS["unknown"])
            if "log2FC" in row and pd.notna(row["log2FC"]):
                title += f"<br>log2FC: {float(row['log2FC']):.3f}"
            if "fdr" in row and pd.notna(row["fdr"]):
                title += f"<br>fdr: {float(row['fdr']):.3g}"
        net.add_node(
            n,
            label=n,
            title=title,
            color=node_color,
            size=float(8.0 + min(35.0, np.sqrt(max(float(deg_w), 0.0)) * 2.4)),
        )

    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "C"))
        color = EDGE_COLORS.get(t, "#8d99ae")
        weight = float(d.get("weight", 1.0))
        sel = float(d.get("selected_value", np.nan))
        title = f"type={t}<br>weight={weight:.4f}"
        if np.isfinite(sel):
            title += f"<br>selected_value={sel:.4f}"
        net.add_edge(u, v, value=max(0.1, weight), color=color, title=title)

    net.set_options(
        """
        var options = {
          "physics": {"enabled": true, "stabilization": {"iterations": 300}},
          "interaction": {"hover": true, "navigationButtons": true},
          "nodes": {"font": {"size": 13}}
        }
        """
    )
    net.write_html(str(out_html), open_browser=False, notebook=False)


def draw_html_single_style(
    g: nx.Graph,
    out_html: Path,
    pair_name: str,
    node_to_comm: dict[str, int],
    template_file: Path,
    interactive_physics: bool,
    interactive_label_mode: str,
) -> bool:
    module = load_single_viz_module()
    if module is None:
        return False
    try:
        return bool(
            module.draw_interactive_html(
                g,
                out_file=out_html,
                title=f"{pair_name}: interactive co-expression network",
                physics=bool(interactive_physics),
                node_to_comm=node_to_comm,
                label_mode=str(interactive_label_mode),
                gene_to_cell_type=None,
                template_file=template_file,
            )
        )
    except Exception:
        return False


def summarize_pair(pair_name: str, g: nx.Graph, out_dir: Path, png: Path, html: Path | None) -> None:
    summary = {
        "pair": pair_name,
        "n_nodes_visualized": int(g.number_of_nodes()),
        "n_edges_visualized": int(g.number_of_edges()),
        "png": str(png),
        "html": str(html) if html is not None else "",
    }
    (out_dir / f"{pair_name}_viz_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def run_pair(pair_dir: Path, out_root: Path, args: argparse.Namespace) -> dict[str, object]:
    pair_name = pair_dir.name
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    edges = read_edges(pair_dir)
    nodes = read_node_stats(pair_dir)
    edges = filter_edges(edges, min_weight=float(args.min_weight), max_edges=int(args.max_edges))
    g = build_graph(edges, min_node_degree=int(args.min_node_degree))

    if not nodes.empty and "gene" in nodes.columns:
        node_attrs = nodes.set_index("gene", drop=False).to_dict(orient="index")
        for node in g.nodes():
            row = node_attrs.get(str(node))
            if row is None:
                continue
            for k, v in row.items():
                if k == "gene":
                    continue
                g.nodes[node][k] = v

    node_to_comm, comm_df = detect_communities(g)
    interactive_template = resolve_base(args.interactive_template)

    edges_csv = pair_out / f"{pair_name}_network_edges.csv"
    nodes_csv = pair_out / f"{pair_name}_network_nodes.csv"
    degree_df = export_edge_node_tables(g, edges_csv, nodes_csv)
    weighted_degree_csv = pair_out / f"{pair_name}_network_weighted_degree.csv"
    degree_df[["gene", "weighted_degree"]].to_csv(weighted_degree_csv, index=False)

    topology_diag, component_df = compute_topology_diagnostics(g, pair_name)
    topology_csv = pair_out / f"{pair_name}_network_topology_metrics.csv"
    pd.DataFrame([topology_diag]).to_csv(topology_csv, index=False)
    component_csv = pair_out / f"{pair_name}_network_component_stats.csv"
    component_df.to_csv(component_csv, index=False)
    community_csv = pair_out / f"{pair_name}_network_community_sizes.csv"
    comm_df.to_csv(community_csv, index=False)

    csd_summary_file = pair_out / f"{pair_name}_csd_feature_summary.csv"
    compute_csd_feature_summary(edges, nodes, pair_name).to_csv(csd_summary_file, index=False)

    png = pair_out / f"{pair_name}_network_global.png"
    draw_png(g, png, pair_name, args)

    hub_png = pair_out / f"{pair_name}_network_hub_subgraph.png"
    draw_hub_subgraph(
        g,
        degree_df,
        hub_png,
        seed=int(args.seed),
        hub_neighbors=40,
        title=pair_name,
    )

    gexf_file = pair_out / f"{pair_name}_network_sparse.gexf"
    graphml_file = pair_out / f"{pair_name}_network_sparse.graphml"
    nx.write_gexf(g, gexf_file)
    nx.write_graphml(g, graphml_file)

    html: Path | None = None
    if bool(args.interactive_html) and HAS_PYVIS:
        html = pair_out / f"{pair_name}_network_interactive.html"
        ok = draw_html_single_style(
            g,
            out_html=html,
            pair_name=pair_name,
            node_to_comm=node_to_comm,
            template_file=interactive_template,
            interactive_physics=bool(args.interactive_physics),
            interactive_label_mode=str(args.interactive_label_mode),
        )
        if not ok:
            draw_html(g, html, pair_name, nodes)

    summarize_pair(pair_name, g, pair_out, png, html)

    print(f"[{pair_name}] visualized nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    return {
        "pair": pair_name,
        "n_nodes_visualized": int(g.number_of_nodes()),
        "n_edges_visualized": int(g.number_of_edges()),
        "png": str(png),
        "hub_png": str(hub_png),
        "html": str(html) if html is not None else "",
        "topology_csv": str(topology_csv),
        "component_csv": str(component_csv),
        "community_csv": str(community_csv),
        "csd_summary_csv": str(csd_summary_file),
    }


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_dirs = list_pair_dirs(in_root, args.pair)
    summaries: list[dict[str, object]] = []
    for pair_dir in pair_dirs:
        summaries.append(run_pair(pair_dir, out_root, args))

    pd.DataFrame(summaries).sort_values("pair").to_csv(
        out_root / "differential_network_viz_index.csv",
        index=False,
    )

    if bool(args.interactive_html) and not HAS_PYVIS:
        print("[warn] pyvis not available; skipping interactive HTML exports")
    print(f"Done. Differential network visualizations: {out_root}")


if __name__ == "__main__":
    main()
