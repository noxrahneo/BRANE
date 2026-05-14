#!/usr/bin/env python3
"""Stage 29 - Degree-preserving topology null tests.

For each single-condition network and each differential network, this script
compares the observed topology to random graphs with the same degree sequence.
The null graphs are sampled with a configuration-model style degree-sequence
generator, then evaluated for modularity, assortativity and clustering.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.network_utils import resolve_base


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results/26_topology_null"

SINGLE_NETWORK_ROOT = REPO_ROOT / "results/13_single_condition_networks"
DIFF_NETWORK_ROOT = REPO_ROOT / "results/14_csd_networks"

SINGLE_NAMES = (
    "Normal",
    "Normal_BRCA1_-_pre-neoplastic",
    "ER_tumor",
    "HER2_tumor",
    "Triple_negative_tumor",
    "Triple_negative_BRCA1_tumor",
)

DIFF_NAMES = (
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate degree-preserving random graphs and compare topology metrics")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--family", choices=("all", "single", "differential"), default="all")
    parser.add_argument("--single-network-dir", default=str(SINGLE_NETWORK_ROOT))
    parser.add_argument("--differential-network-dir", default=str(DIFF_NETWORK_ROOT))
    parser.add_argument("--condition", action="append", default=[], help="Limit to specific single-condition network(s)")
    parser.add_argument("--pair", action="append", default=[], help="Limit to specific differential pair(s)")
    return parser.parse_args()


def _build_graph(edge_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in edge_df.iterrows():
        gene_a = str(row["gene_a"])
        gene_b = str(row["gene_b"])
        if gene_a and gene_b and gene_a != gene_b:
            g.add_edge(gene_a, gene_b)
    return g


def _observed_metrics(g: nx.Graph, seed: int) -> dict[str, float]:
    if g.number_of_nodes() < 2 or g.number_of_edges() == 0:
        return {
            "n_nodes": int(g.number_of_nodes()),
            "n_edges": int(g.number_of_edges()),
            "modularity_q": float("nan"),
            "assortativity_coefficient": float("nan"),
            "average_clustering_coefficient": float("nan"),
        }

    try:
        nodes = list(g.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        ig_g = ig.Graph(
            n=len(nodes),
            edges=[(node_idx[u], node_idx[v]) for u, v in g.edges()],
        )
        partition = leidenalg.find_partition(
            ig_g, leidenalg.ModularityVertexPartition, seed=seed
        )
        communities = [frozenset(nodes[i] for i in c) for c in partition]
        modularity_q = float(nx.community.modularity(g, communities))
    except Exception:
        modularity_q = float("nan")

    try:
        assortativity = float(nx.degree_assortativity_coefficient(g))
    except Exception:
        assortativity = float("nan")

    try:
        clustering = float(nx.average_clustering(g))
    except Exception:
        clustering = float("nan")

    return {
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "modularity_q": modularity_q,
        "assortativity_coefficient": assortativity,
        "average_clustering_coefficient": clustering,
    }


def _sample_degree_sequence_graph(degrees: list[int], seed: int) -> nx.Graph:
    degrees = [int(max(0, d)) for d in degrees]
    multigraph = nx.configuration_model(degrees, seed=seed)
    g = nx.Graph(multigraph)
    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_nodes_from(range(len(degrees)))

    if g.number_of_nodes() != len(degrees):
        g.add_nodes_from(range(len(degrees)))
    return g


def _empirical_upper_tail(null_values: np.ndarray, observed: float) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    hits = int(np.sum(null_values >= observed))
    return float((hits + 1.0) / (null_values.size + 1.0))


def _edge_file(root: Path, name: str, family: str) -> Path:
    if family == "single":
        return root / name / f"{name}_edges.tsv"
    return root / name / f"{name}_differential_edges_permutation.csv"


def _iter_networks(root: Path, family: str, keep: set[str]) -> list[tuple[str, Path]]:
    names = SINGLE_NAMES if family == "single" else DIFF_NAMES
    out: list[tuple[str, Path]] = []
    for name in names:
        if keep and name not in keep:
            continue
        edge_file = _edge_file(root, name, family)
        if edge_file.exists():
            out.append((name, edge_file))
        else:
            log.warning("Missing edge file for %s/%s: %s", family, name, edge_file)
    return out


def _read_edges(edge_file: Path) -> pd.DataFrame:
    if edge_file.suffix == ".tsv":
        return pd.read_csv(edge_file, sep="\t")
    return pd.read_csv(edge_file)


def _run_network(name: str, family: str, edge_file: Path, n_null: int, seed: int, out_dir: Path) -> dict[str, object]:
    edge_df = _read_edges(edge_file)
    g_obs = _build_graph(edge_df)
    observed = _observed_metrics(g_obs, seed=seed)

    degrees = [int(d) for _, d in g_obs.degree()]
    if not degrees:
        raise ValueError(f"No nodes detected for {family}/{name}")

    null_rows: list[dict[str, float]] = []
    for i in tqdm(range(int(n_null)), desc=f"{family}:{name}", unit="graph", leave=False):
        null_seed = seed + i + 1
        g_null = _sample_degree_sequence_graph(degrees, seed=null_seed)
        null_rows.append({"iteration": i + 1, **_observed_metrics(g_null, seed=null_seed)})

    null_df = pd.DataFrame(null_rows)
    null_file = out_dir / f"{name}_null_metrics.csv"
    null_df.to_csv(null_file, index=False)

    summary = {
        "family": family,
        "name": name,
        "edge_file": str(edge_file),
        "null_metrics_file": str(null_file),
        "n_nodes": int(observed["n_nodes"]),
        "n_edges": int(observed["n_edges"]),
        "n_null": int(n_null),
        "seed": int(seed),
    }

    for metric_key in ("modularity_q", "assortativity_coefficient", "average_clustering_coefficient"):
        null_vals = pd.to_numeric(null_df[metric_key], errors="coerce").to_numpy(dtype=float)
        summary[f"observed_{metric_key}"] = float(observed[metric_key])
        summary[f"null_mean_{metric_key}"] = float(np.nanmean(null_vals)) if np.isfinite(null_vals).any() else float("nan")
        summary[f"null_sd_{metric_key}"] = float(np.nanstd(null_vals, ddof=1)) if np.isfinite(null_vals).sum() > 1 else float("nan")
        summary[f"null_q95_{metric_key}"] = float(np.nanquantile(null_vals, 0.95)) if np.isfinite(null_vals).any() else float("nan")
        summary[f"p_empirical_{metric_key}"] = _empirical_upper_tail(null_vals, float(observed[metric_key]))

    return summary


def main() -> int:
    args = parse_args()
    output_dir = resolve_base(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_root = resolve_base(args.single_network_dir)
    diff_root = resolve_base(args.differential_network_dir)

    keep_conditions = set(args.condition)
    keep_pairs = set(args.pair)

    summaries: list[dict[str, object]] = []

    family_specs = []
    if args.family in ("all", "single"):
        family_specs.append(("single", single_root, keep_conditions))
    if args.family in ("all", "differential"):
        family_specs.append(("differential", diff_root, keep_pairs))

    for family, root, keep in family_specs:
        networks = _iter_networks(root=root, family=family, keep=keep)
        if not networks:
            continue

        family_out = output_dir / family
        family_out.mkdir(parents=True, exist_ok=True)

        for name, edge_file in networks:
            log.info("Running topology null for %s/%s", family, name)
            summaries.append(
                _run_network(
                    name=name,
                    family=family,
                    edge_file=edge_file,
                    n_null=int(args.n_null),
                    seed=int(args.seed),
                    out_dir=family_out,
                )
            )

    if not summaries:
        log.warning("No networks were processed")
        return 0

    summary_df = pd.DataFrame(summaries).sort_values(["family", "name"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "topology_null_summary.csv", index=False)
    log.info("Wrote topology-null summary to %s", output_dir / "topology_null_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())