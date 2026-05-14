#!/usr/bin/env python3
# flake8: noqa: E501
"""Visualize stage-09 persistent overlap networks (edges present in both branch A and B)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from pyvis.network import Network  # type: ignore[import-not-found]

    HAS_PYVIS = True
except Exception:
    Network = None  # type: ignore[assignment]
    HAS_PYVIS = False

from utils.network_utils import resolve_base

EDGE_COLORS = {"C": "#2a9d8f", "S": "#f4a261", "D": "#e76f51", "mixed": "#8d99ae"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize stage-09 persistent overlap networks")
    p.add_argument("--branch-a-dir", default="results/15_scalefree_networks")
    p.add_argument("--branch-b-dir", default="results/17_permutation_networks")
    p.add_argument("--output-dir", default="results/19_persistent_overlap")
    p.add_argument("--pair", action="append", default=[], help="Pair folder name case__vs__control. Repeat for multiple")
    p.add_argument(
        "--mode",
        choices=["persistent_edges", "shared_hub_core"],
        default="persistent_edges",
        help="persistent_edges (recommended): all A∩B edges; shared_hub_core: only persistent edges among shared top hubs",
    )
    p.add_argument("--top-hubs", type=int, default=50, help="Top hubs per branch used for shared_hub_core mode")
    p.add_argument("--viz-top-k", type=int, default=0, help="Visualization-only top-K overlap edges (0 means no limit)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--interactive-html", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--interactive-template", default="scripts/network_interactive_template.html")
    p.add_argument("--interactive-physics", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--interactive-label-mode", choices=["hubs", "all", "none"], default="hubs")
    return p.parse_args()


def edge_file_for(branch_dir: Path, pair_name: str) -> Path:
    a = branch_dir / pair_name / f"{pair_name}_differential_edges_scalefree.csv"
    b = branch_dir / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    if a.exists():
        return a
    if b.exists():
        return b
    raise FileNotFoundError(f"Missing branch edge file in {branch_dir / pair_name}")


def hubs_file_for(branch_dir: Path, pair_name: str) -> Path:
    a = branch_dir / pair_name / f"{pair_name}_top_hubs_scalefree.csv"
    b = branch_dir / pair_name / f"{pair_name}_top_hubs_permutation.csv"
    if a.exists():
        return a
    if b.exists():
        return b
    raise FileNotFoundError(f"Missing branch hub file in {branch_dir / pair_name}")


def load_shared_hubs(pair_name: str, a_root: Path, b_root: Path, top_hubs: int) -> set[str]:
    a_hubs = pd.read_csv(hubs_file_for(a_root, pair_name)).head(int(top_hubs))
    b_hubs = pd.read_csv(hubs_file_for(b_root, pair_name)).head(int(top_hubs))
    genes_a = set(a_hubs["gene"].astype(str).tolist()) if not a_hubs.empty and "gene" in a_hubs.columns else set()
    genes_b = set(b_hubs["gene"].astype(str).tolist()) if not b_hubs.empty and "gene" in b_hubs.columns else set()
    return genes_a.intersection(genes_b)


def canonical_key(gene_a: str, gene_b: str) -> tuple[str, str]:
    a = str(gene_a)
    b = str(gene_b)
    return (a, b) if a <= b else (b, a)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _best_value(primary: float, fallback: float) -> float:
    if np.isfinite(primary):
        return float(primary)
    if np.isfinite(fallback):
        return float(fallback)
    return float("nan")


def _combine_persist_score(a_val: float, b_val: float, a_w: float, b_w: float) -> float:
    va = _best_value(a_val, a_w)
    vb = _best_value(b_val, b_w)
    if np.isfinite(va) and np.isfinite(vb):
        return float(min(va, vb))
    if np.isfinite(va):
        return float(va)
    if np.isfinite(vb):
        return float(vb)
    return float("nan")


def build_overlap_edges(edges_a: pd.DataFrame, edges_b: pd.DataFrame) -> pd.DataFrame:
    if edges_a.empty or edges_b.empty:
        return pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "weight",
                "persist_score",
                "selected_value_a",
                "selected_value_b",
                "link_type",
                "link_type_a",
                "link_type_b",
                "wTO",
                "wTO_a",
                "wTO_b",
            ]
        )

    a_map: dict[tuple[str, str], dict[str, object]] = {}
    b_map: dict[tuple[str, str], dict[str, object]] = {}

    for row in edges_a.to_dict(orient="records"):
        row_dict = {str(k): v for k, v in row.items()}
        key = canonical_key(str(row_dict.get("gene_a", "")), str(row_dict.get("gene_b", "")))
        a_map[key] = row_dict

    for row in edges_b.to_dict(orient="records"):
        row_dict = {str(k): v for k, v in row.items()}
        key = canonical_key(str(row_dict.get("gene_a", "")), str(row_dict.get("gene_b", "")))
        b_map[key] = row_dict

    shared_keys = sorted(set(a_map.keys()).intersection(b_map.keys()))
    rows: list[dict[str, object]] = []
    for key in shared_keys:
        ra = a_map[key]
        rb = b_map[key]

        sel_a = to_float(ra.get("selected_value", np.nan))
        sel_b = to_float(rb.get("selected_value", np.nan))
        w_a = to_float(ra.get("weight", np.nan))
        w_b = to_float(rb.get("weight", np.nan))

        persist_score = _combine_persist_score(sel_a, sel_b, w_a, w_b)
        weight = persist_score if np.isfinite(persist_score) else _combine_persist_score(w_a, w_b, np.nan, np.nan)

        l_a = str(ra.get("link_type", "mixed"))
        l_b = str(rb.get("link_type", "mixed"))
        l_shared = l_a if l_a == l_b else "mixed"

        wto_a = to_float(ra.get("wTO", np.nan))
        wto_b = to_float(rb.get("wTO", np.nan))
        if np.isfinite(wto_a) and np.isfinite(wto_b):
            wto = float((wto_a + wto_b) / 2.0)
        elif np.isfinite(wto_a):
            wto = float(wto_a)
        elif np.isfinite(wto_b):
            wto = float(wto_b)
        else:
            wto = float("nan")

        rows.append(
            {
                "gene_a": key[0],
                "gene_b": key[1],
                "weight": weight,
                "persist_score": persist_score,
                "selected_value_a": sel_a,
                "selected_value_b": sel_b,
                "link_type": l_shared,
                "link_type_a": l_a,
                "link_type_b": l_b,
                "wTO": wto,
                "wTO_a": wto_a,
                "wTO_b": wto_b,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_sort_score"] = pd.to_numeric(out["persist_score"], errors="coerce")
    out["_sort_weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.sort_values(["_sort_score", "_sort_weight"], ascending=False).drop(columns=["_sort_score", "_sort_weight"]).reset_index(drop=True)
    return out


def subset_edges_for_viz(edges_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df.copy()
    work = edges_df.copy()
    work["persist_score"] = pd.to_numeric(work["persist_score"], errors="coerce")
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work = work.sort_values(["persist_score", "weight"], ascending=False)
    if int(top_k) > 0:
        work = work.head(int(top_k)).copy().reset_index(drop=True)
    return work


def build_graph(edges_df: pd.DataFrame) -> nx.Graph:
    if edges_df.empty:
        return nx.Graph()
    g = nx.Graph()
    for row in edges_df.to_dict(orient="records"):
        g.add_edge(
            str(row["gene_a"]),
            str(row["gene_b"]),
            weight=to_float(row.get("weight", 1.0)),
            link_type=str(row.get("link_type", "mixed")),
            persist_score=to_float(row.get("persist_score", np.nan)),
            wTO=to_float(row.get("wTO", np.nan)),
        )
    return g


def draw_png(g: nx.Graph, out_png: Path, title: str, dpi: int, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No shared edges", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(g, seed=int(seed), weight="weight", iterations=180)
    deg_w = dict(g.degree(weight="weight"))
    vals = np.array(list(deg_w.values()), dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        sizes = {k: 110.0 for k in deg_w}
    else:
        sizes = {k: float(80.0 + 360.0 * ((v - vmin) / (vmax - vmin))) for k, v in deg_w.items()}

    edge_groups: dict[str, list[tuple[str, str]]] = {"C": [], "S": [], "D": [], "mixed": [], "other": []}
    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "other"))
        if t not in edge_groups:
            t = "other"
        edge_groups[t].append((u, v))

    for t, edges in edge_groups.items():
        if not edges:
            continue
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edges,
            edge_color=EDGE_COLORS.get(t, "#8d99ae"),
            width=0.9,
            alpha=0.6,
            ax=ax,
        )

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=[sizes.get(n, 100.0) for n in g.nodes()],
        node_color="#457b9d",
        alpha=0.88,
        linewidths=0.25,
        edgecolors="#f1faee",
        ax=ax,
    )

    top_nodes = sorted(g.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:20]
    labels = {n: n for n, _ in top_nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="#1d3557", ax=ax)

    ax.set_title(f"{title} | nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    ax.axis("off")
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_html(g: nx.Graph, out_html: Path, title: str) -> None:
    if not HAS_PYVIS or Network is None:
        return
    net: Any = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#1f2937")
    net.barnes_hut(gravity=-12000, central_gravity=0.2, spring_length=170, spring_strength=0.02)

    for n, d_w in g.degree(weight="weight"):
        node_title = f"gene: {n}<br>weighted_degree: {d_w:.3f}"
        net.add_node(
            n,
            label=n,
            title=node_title,
            color="#457b9d",
            size=float(8.0 + min(35.0, np.sqrt(max(float(d_w), 0.0)) * 2.4)),
        )

    for u, v, d in g.edges(data=True):
        t = str(d.get("link_type", "mixed"))
        w = to_float(d.get("weight", 1.0))
        ps = to_float(d.get("persist_score", np.nan))
        edge_title = f"type={t}<br>weight={w:.4f}"
        if np.isfinite(ps):
            edge_title += f"<br>persist_score={ps:.4f}"
        net.add_edge(u, v, value=max(0.1, w), color=EDGE_COLORS.get(t, "#8d99ae"), title=edge_title)

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


def load_single_viz_module(base_root: Path):
    script_path = base_root / "scripts" / "26_networkx_visualization.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("single_viz26", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_communities(g: nx.Graph) -> dict[str, int]:
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return {}
    communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    communities = sorted(communities, key=len, reverse=True)
    node_to_comm: dict[str, int] = {}
    for idx, comm_nodes in enumerate(communities, start=1):
        for node in comm_nodes:
            node_to_comm[str(node)] = int(idx)
    return node_to_comm


def draw_html_single_style(
    g: nx.Graph,
    out_html: Path,
    title: str,
    node_to_comm: dict[str, int],
    template_file: Path,
    interactive_physics: bool,
    interactive_label_mode: str,
    base_root: Path,
) -> bool:
    module = load_single_viz_module(base_root)
    if module is None:
        return False
    try:
        return bool(
            module.draw_interactive_html(
                g,
                out_file=out_html,
                title=title,
                physics=bool(interactive_physics),
                node_to_comm=node_to_comm,
                label_mode=str(interactive_label_mode),
                gene_to_cell_type=None,
                template_file=template_file,
            )
        )
    except Exception:
        return False


def run_pair(pair_name: str, a_root: Path, b_root: Path, out_root: Path, args: argparse.Namespace) -> dict[str, object]:
    edge_a_file = edge_file_for(a_root, pair_name)
    edge_b_file = edge_file_for(b_root, pair_name)
    edges_a = pd.read_csv(edge_a_file)
    edges_b = pd.read_csv(edge_b_file)

    overlap_edges = build_overlap_edges(edges_a, edges_b)
    n_overlap_source = int(overlap_edges.shape[0])

    shared_hubs = load_shared_hubs(pair_name, a_root, b_root, int(args.top_hubs)) if args.mode == "shared_hub_core" else set()
    if args.mode == "shared_hub_core":
        overlap_edges = overlap_edges[
            overlap_edges["gene_a"].astype(str).isin(shared_hubs)
            & overlap_edges["gene_b"].astype(str).isin(shared_hubs)
        ].copy()
        overlap_edges = overlap_edges.reset_index(drop=True)

    n_overlap_mode = int(overlap_edges.shape[0])
    viz_edges = subset_edges_for_viz(overlap_edges, top_k=int(args.viz_top_k))
    g = build_graph(viz_edges)

    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    mode_tag = str(args.mode)

    overlap_csv = pair_out / f"{pair_name}_{mode_tag}_edges.csv"
    viz_csv = pair_out / f"{pair_name}_{mode_tag}_edges_viz.csv"
    overlap_edges.to_csv(overlap_csv, index=False)
    viz_edges.to_csv(viz_csv, index=False)

    png = pair_out / f"{pair_name}_{mode_tag}_network_global.png"
    draw_png(g, png, f"{pair_name} [{mode_tag} A∩B]", dpi=int(args.dpi), seed=int(args.seed))

    html = None
    if bool(args.interactive_html):
        html = pair_out / f"{pair_name}_{mode_tag}_network_interactive.html"
        template_file = resolve_base(args.interactive_template)
        node_to_comm = detect_communities(g)
        ok = draw_html_single_style(
            g,
            out_html=html,
            title=f"{pair_name}: {mode_tag} network [A∩B]",
            node_to_comm=node_to_comm,
            template_file=template_file,
            interactive_physics=bool(args.interactive_physics),
            interactive_label_mode=str(args.interactive_label_mode),
            base_root=REPO_ROOT,
        )
        if not ok and HAS_PYVIS:
            draw_html(g, html, f"{pair_name} [{mode_tag} A∩B]")

    summary = {
        "pair": pair_name,
        "mode": mode_tag,
        "source_edge_file_a": str(edge_a_file),
        "source_edge_file_b": str(edge_b_file),
        "overlap_edges_file": str(overlap_csv),
        "overlap_edges_viz_file": str(viz_csv),
        "top_hubs": int(args.top_hubs),
        "n_shared_hub_genes": int(len(shared_hubs)) if args.mode == "shared_hub_core" else None,
        "viz_top_k": int(args.viz_top_k),
        "n_edges_overlap_source": int(n_overlap_source),
        "n_edges_overlap_mode": int(n_overlap_mode),
        "n_nodes_visualized": int(g.number_of_nodes()),
        "n_edges_visualized": int(g.number_of_edges()),
        "png": str(png),
        "html": str(html) if html is not None else "",
    }
    (pair_out / f"{pair_name}_{mode_tag}_viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{pair_name}][{mode_tag}] nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
        f"(source_overlap={n_overlap_source}, mode_overlap={n_overlap_mode})"
    )
    return summary


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
        raise ValueError("No overlapping pair folders found between branch A and branch B")

    rows: list[dict[str, object]] = []
    for pair_name in pairs:
        rows.append(run_pair(pair_name, a_root, b_root, out_root, args))

    mode_tag = str(args.mode)
    pd.DataFrame(rows).sort_values(["pair"]).to_csv(out_root / f"{mode_tag}_viz_index.csv", index=False)
    if bool(args.interactive_html) and not HAS_PYVIS:
        print("[warn] pyvis not available; skipping interactive HTML exports")
    print(f"Done. Stage-09 persistent-overlap visualizations ({mode_tag}): {out_root}")


if __name__ == "__main__":
    main()
