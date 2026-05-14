#!/usr/bin/env python3
"""Build compact network indexes and aggregated co-expression tables.

Generates, for each mode in {single, combined}:
- network_index.csv / network_index.md
- network_edges_all.csv
- network_nodes_all.csv
- network_communities_all.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact indexes and aggregated co-expression CSVs"
    )
    parser.add_argument(
        "--prep-root",
        default="results/11_coexpression_prep",
        help="Root containing single/ and combined/ prep outputs",
    )
    parser.add_argument(
        "--viz-root",
        default="results/13_coexpression_viz",
        help="Root containing single/ and combined/ visualization outputs",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "combined", "both"],
        default="both",
        help="Which subfolders to process",
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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_condition_dirs(viz_mode_dir: Path) -> list[Path]:
    if not viz_mode_dir.exists():
        return []
    cond_dirs: list[Path] = []
    for p in sorted(viz_mode_dir.iterdir()):
        if not p.is_dir():
            continue
        summary = p / f"{p.name}_network_viz_summary.json"
        if summary.exists():
            cond_dirs.append(p)
    return cond_dirs


def detect_network_type_dirs(*mode_dirs: Path) -> list[str]:
    known = ["signed", "unsigned"]
    found: list[str] = []
    for nt in known:
        for mode_dir in mode_dirs:
            if (mode_dir / nt).is_dir():
                found.append(nt)
                break
    return found


def module_rows_from_edges_df(
    edges_df: pd.DataFrame,
    condition: str,
    source_conditions: list[str],
    selected_power: int | None,
) -> list[dict]:
    if edges_df.empty:
        return []

    required_cols = {"gene_a", "gene_b", "weight"}
    if not required_cols.issubset(set(edges_df.columns)):
        return []

    g = nx.Graph()
    for row in edges_df.itertuples(index=False):
        try:
            gene_a = str(getattr(row, "gene_a"))
            gene_b = str(getattr(row, "gene_b"))
            weight = float(getattr(row, "weight"))
        except Exception:
            continue
        g.add_edge(gene_a, gene_b, weight=weight)

    if g.number_of_nodes() == 0:
        return []

    communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    communities = sorted(communities, key=len, reverse=True)
    weighted_degree = dict(g.degree(weight="weight"))
    src_text = " | ".join(source_conditions)

    rows: list[dict] = []
    for community_id, comm_nodes in enumerate(communities, start=1):
        comm = set(str(n) for n in comm_nodes)
        sub = g.subgraph(comm)
        weights = [float(d.get("weight", 0.0)) for _, _, d in sub.edges(data=True)]
        mean_weight = float(sum(weights) / len(weights)) if weights else 0.0

        hubs = sorted(comm, key=lambda n: weighted_degree.get(n, 0.0), reverse=True)
        top_hubs = hubs[:3]

        rows.append(
            {
                "condition": condition,
                "source_conditions": src_text,
                "selected_power": selected_power,
                "community_id": int(community_id),
                "n_nodes": int(len(comm)),
                "n_internal_edges": int(sub.number_of_edges()),
                "mean_internal_edge_weight": mean_weight,
                "top_hub_genes": " | ".join(top_hubs),
                "top_hub_1": top_hubs[0] if len(top_hubs) > 0 else "",
                "top_hub_2": top_hubs[1] if len(top_hubs) > 1 else "",
                "top_hub_3": top_hubs[2] if len(top_hubs) > 2 else "",
            }
        )

    return rows


def get_selected_power_info(prep_mode_dir: Path, condition: str) -> tuple[int | None, list[str]]:
    selected_json = prep_mode_dir / condition / f"{condition}_selected_power.json"
    meta = load_json(selected_json)
    selected_power = meta.get("selected_power")
    source_conditions = meta.get("source_conditions", [])
    if isinstance(source_conditions, list):
        src = [str(x) for x in source_conditions]
    else:
        src = []
    try:
        selected_power = int(selected_power) if selected_power is not None else None
    except Exception:
        selected_power = None
    return selected_power, src


def aggregate_mode(prep_mode_dir: Path, viz_mode_dir: Path) -> None:
    cond_dirs = iter_condition_dirs(viz_mode_dir)

    index_rows: list[dict] = []
    edges_all: list[pd.DataFrame] = []
    nodes_all: list[pd.DataFrame] = []
    comm_all: list[pd.DataFrame] = []
    module_rows: list[dict] = []

    for cond_dir in cond_dirs:
        condition = cond_dir.name
        summary_path = cond_dir / f"{condition}_network_viz_summary.json"
        summary = load_json(summary_path)

        selected_power, source_conditions = get_selected_power_info(
            prep_mode_dir=prep_mode_dir,
            condition=condition,
        )

        n_nodes = int(summary.get("n_nodes", 0) or 0)
        n_edges = int(summary.get("n_edges", 0) or 0)
        density = float((2 * n_edges) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

        row = {
            "condition": condition,
            "selected_power": selected_power,
            "source_conditions": " | ".join(source_conditions),
            "n_source_conditions": len(source_conditions),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "network_type": summary.get("network_type", ""),
            "min_weight": summary.get("min_weight", ""),
            "max_edges_requested": summary.get("max_edges_requested", ""),
            "interactive_html_generated": bool(summary.get("interactive_html_generated", False)),
            "summary_json": str(summary_path),
        }
        index_rows.append(row)

        edges_file = cond_dir / f"{condition}_network_edges.csv"
        if edges_file.exists():
            try:
                edf = pd.read_csv(edges_file)
                module_rows.extend(
                    module_rows_from_edges_df(
                        edges_df=edf,
                        condition=condition,
                        source_conditions=source_conditions,
                        selected_power=selected_power,
                    )
                )
                edf.insert(0, "condition", condition)
                if source_conditions:
                    edf.insert(1, "source_conditions", " | ".join(source_conditions))
                edges_all.append(edf)
            except Exception:
                pass

        nodes_file = cond_dir / f"{condition}_network_nodes.csv"
        if nodes_file.exists():
            try:
                ndf = pd.read_csv(nodes_file)
                ndf.insert(0, "condition", condition)
                if source_conditions:
                    ndf.insert(1, "source_conditions", " | ".join(source_conditions))
                nodes_all.append(ndf)
            except Exception:
                pass

        comm_file = cond_dir / f"{condition}_network_community_sizes.csv"
        if comm_file.exists():
            try:
                cdf = pd.read_csv(comm_file)
                cdf.insert(0, "condition", condition)
                if source_conditions:
                    cdf.insert(1, "source_conditions", " | ".join(source_conditions))
                comm_all.append(cdf)
            except Exception:
                pass

    index_df = pd.DataFrame(index_rows).sort_values(["n_edges", "n_nodes"], ascending=[False, False])
    index_csv = viz_mode_dir / "network_index.csv"
    index_df.to_csv(index_csv, index=False)

    md_df = index_df.copy()
    if "density" in md_df.columns:
        md_df["density"] = md_df["density"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    md_cols = [
        "condition",
        "selected_power",
        "n_source_conditions",
        "n_nodes",
        "n_edges",
        "density",
        "network_type",
        "interactive_html_generated",
    ]
    md_cols = [c for c in md_cols if c in md_df.columns]
    header = "| " + " | ".join(md_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(md_cols)) + " |"
    lines = [header, sep]
    for _, row in md_df[md_cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in md_cols) + " |")
    index_md = viz_mode_dir / "network_index.md"
    index_md.write_text("# Network index\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

    if edges_all:
        pd.concat(edges_all, ignore_index=True).to_csv(viz_mode_dir / "network_edges_all.csv", index=False)
    if nodes_all:
        pd.concat(nodes_all, ignore_index=True).to_csv(viz_mode_dir / "network_nodes_all.csv", index=False)
    if comm_all:
        pd.concat(comm_all, ignore_index=True).to_csv(viz_mode_dir / "network_communities_all.csv", index=False)
    if module_rows:
        module_df = pd.DataFrame(module_rows).sort_values(
            ["condition", "community_id"],
            ascending=[True, True],
        )
        module_df.to_csv(viz_mode_dir / "module_condition_associations.csv", index=False)

    print(f"[{viz_mode_dir.name}] wrote index: {index_csv}")
    print(f"[{viz_mode_dir.name}] wrote index: {index_md}")
    if edges_all:
        print(f"[{viz_mode_dir.name}] wrote: {viz_mode_dir / 'network_edges_all.csv'}")
    if nodes_all:
        print(f"[{viz_mode_dir.name}] wrote: {viz_mode_dir / 'network_nodes_all.csv'}")
    if comm_all:
        print(f"[{viz_mode_dir.name}] wrote: {viz_mode_dir / 'network_communities_all.csv'}")
    if module_rows:
        print(f"[{viz_mode_dir.name}] wrote: {viz_mode_dir / 'module_condition_associations.csv'}")


def main() -> None:
    args = parse_args()
    prep_root = resolve_base(args.prep_root)
    viz_root = resolve_base(args.viz_root)

    modes = ["single", "combined"] if args.mode == "both" else [args.mode]
    for mode in modes:
        prep_mode_dir = prep_root / mode
        viz_mode_dir = viz_root / mode
        viz_mode_dir.mkdir(parents=True, exist_ok=True)
        network_type_dirs = detect_network_type_dirs(prep_mode_dir, viz_mode_dir)

        if network_type_dirs:
            for network_type in network_type_dirs:
                prep_type_dir = prep_mode_dir / network_type
                viz_type_dir = viz_mode_dir / network_type
                viz_type_dir.mkdir(parents=True, exist_ok=True)
                aggregate_mode(
                    prep_mode_dir=prep_type_dir,
                    viz_mode_dir=viz_type_dir,
                )
        else:
            aggregate_mode(prep_mode_dir=prep_mode_dir, viz_mode_dir=viz_mode_dir)


if __name__ == "__main__":
    main()
