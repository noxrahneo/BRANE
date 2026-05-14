#!/usr/bin/env python3
"""
Stage 14 — Network Tag Permutation.

Tests whether cell-type labels are spatially clustered on the persistent
co-expression network beyond what is expected by chance.

Metric: weighted neighbourhood homophily
  For each tagged node v with cell type t:
    h(v) = sum_{u in tagged_neighbours(v)} w(v,u) * 1[tag(u)==t]
           / sum_{u in tagged_neighbours(v)} w(v,u)

Null model: label-preserving Fisher-Yates shuffle of cell_type labels
  among labelled nodes only. Preserves per-type counts.

Nodes with no cell_type label are excluded from both the metric and the
shuffle pool but remain in the network topology.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

TAGGING_DIR = REPO_ROOT / "results/20_node_annotation/03_output_with_lfc"
CSD_NETWORKS_DIR = REPO_ROOT / "results/14_csd_networks"
OUTPUT_ROOT = (
    REPO_ROOT
    / "results/22_homophily"
)

PAIR_NAMES = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

# Pairs where small network warrants an interpretive note
SPARSE_PAIRS = {"Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic"}

MIN_NODES_PER_TYPE = 5   # minimum labelled nodes to report per-type result
DEFAULT_N_PERM = 500


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_viz_inputs(pair_name: str) -> tuple[Path, Path]:
    node_path = TAGGING_DIR / f"{pair_name}_tagged_with_lfc.csv"
    edge_path = CSD_NETWORKS_DIR / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    if not node_path.exists():
        raise FileNotFoundError(f"Node file not found: {node_path}")
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    return node_path, edge_path


def load_pair(pair_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_path, edge_path = _resolve_viz_inputs(pair_name)
    nodes = pd.read_csv(node_path, low_memory=False)
    edges = pd.read_csv(edge_path, low_memory=False)
    return nodes, edges


# ---------------------------------------------------------------------------
# Graph construction (adjacency over tagged nodes)
# ---------------------------------------------------------------------------

def build_weighted_adjacency(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    tag_col: str = "cell_type",
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """
    Return:
      adj  — {gene: {neighbour_gene: weight}} for all edge endpoints
      tags — {gene: cell_type} for labelled nodes only
    """
    symbol_col = "approved_symbol" if "approved_symbol" in nodes.columns else "gene"

    # Build gene → cell_type map (drop unlabelled)
    tag_series = (
        nodes[["gene", tag_col]]
        .dropna(subset=[tag_col])
        .drop_duplicates(subset=["gene"])
        .set_index("gene")[tag_col]
    )
    tags: dict[str, str] = tag_series.to_dict()

    # Build adjacency from edges (weighted, undirected)
    adj: dict[str, dict[str, float]] = {}
    for _, row in edges.iterrows():
        a, b = str(row["gene_a"]), str(row["gene_b"])
        w = float(row["weight"]) if not np.isnan(float(row["weight"])) else 0.0
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w

    return adj, tags


# ---------------------------------------------------------------------------
# Homophily computation
# ---------------------------------------------------------------------------

def _node_homophily(
    gene: str,
    tag: str,
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
) -> float | None:
    """Weighted homophily for a single node. Returns None if no tagged neighbours."""
    neighbours = adj.get(gene, {})
    total_w = 0.0
    same_w = 0.0
    for nbr, w in neighbours.items():
        if nbr in tags:
            total_w += w
            if tags[nbr] == tag:
                same_w += w
    if total_w == 0.0:
        return None
    return same_w / total_w


def compute_homophily(
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
) -> tuple[float | None, dict[str, float]]:
    """
    Compute overall homophily and per-cell-type homophily.

    Returns:
        overall  — mean h(v) across all eligible labelled nodes
        per_type — {cell_type: mean h(v) for nodes of that type}
    """
    per_type_vals: dict[str, list[float]] = {}
    all_vals: list[float] = []

    for gene, tag in tags.items():
        h = _node_homophily(gene, tag, adj, tags)
        if h is not None:
            all_vals.append(h)
            per_type_vals.setdefault(tag, []).append(h)

    overall = float(np.mean(all_vals)) if all_vals else None
    per_type = {t: float(np.mean(vs)) for t, vs in per_type_vals.items()}
    return overall, per_type


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

def run_permutations(
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[list[float], dict[str, list[float]]]:
    """
    Perform label-preserving permutations.

    Returns:
        null_overall   — list of overall H values from each permutation
        null_per_type  — {cell_type: list of H_t values}
    """
    genes = list(tags.keys())
    labels = list(tags.values())
    all_types = set(labels)

    null_overall: list[float] = []
    null_per_type: dict[str, list[float]] = {t: [] for t in all_types}

    for _ in range(n_perm):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        perm_tags = dict(zip(genes, shuffled))
        h_overall, h_per_type = compute_homophily(adj, perm_tags)
        if h_overall is not None:
            null_overall.append(h_overall)
        for t in all_types:
            if t in h_per_type:
                null_per_type[t].append(h_per_type[t])

    return null_overall, null_per_type


def plot_convergence(
    pair_name: str,
    null_overall: list[float],
    real_overall: float | None,
    out_path: Path,
) -> None:
    """
    Plot the running (cumulative) mean and ±1 SD of the null distribution
    as permutations accumulate. A flat line means 500 permutations was enough.
    """
    arr = np.array(null_overall)
    n = len(arr)
    steps = np.arange(1, n + 1)
    running_mean = np.cumsum(arr) / steps
    running_std = np.array([arr[:i].std(ddof=1) if i > 1 else 0.0 for i in steps])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(steps, running_mean, color="#4c72b0", linewidth=1.5, label="Running mean (null)")
    ax.fill_between(
        steps,
        running_mean - running_std,
        running_mean + running_std,
        alpha=0.25,
        color="#4c72b0",
        label="±1 SD",
    )
    ax.set_xlabel("Number of permutations", fontsize=10)
    ax.set_ylabel("Homophily (H)", fontsize=10)
    ax.set_title(f"{pair_name.replace('__vs__', ' vs ')} — null convergence", fontsize=10)
    # Y axis: scale to null distribution range only
    y_min = max(0.0, (running_mean - running_std).min() * 0.95)
    y_max = (running_mean + running_std).max() * 1.05
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(real: float, null: list[float]) -> dict:
    null_arr = np.array(null)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else 0.0
    z = (real - null_mean) / null_std if null_std > 0 else float("nan")
    p = float(np.mean(null_arr >= real))
    return {
        "real_H": real,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z,
        "empirical_p": p,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_null_distributions(
    pair_name: str,
    real_overall: float | None,
    null_overall: list[float],
    real_per_type: dict[str, float],
    null_per_type: dict[str, list[float]],
    tags: dict[str, str],
    min_nodes: int,
    out_path: Path,
) -> None:
    from collections import Counter
    type_counts = Counter(tags.values())
    eligible = sorted(
        [t for t, c in type_counts.items() if c >= min_nodes],
        key=lambda t: -type_counts[t],
    )

    n_panels = 1 + len(eligible)   # overall + one per eligible type
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    def _draw(ax: plt.Axes, null: list[float], real: float | None, title: str, n: int) -> None:
        if null:
            ax.hist(null, bins=40, color="#6baed6", edgecolor="white", linewidth=0.3, alpha=0.85)
        if real is not None:
            ax.axvline(real, color="#d62728", linewidth=2, label=f"Observed H={real:.3f}")
            ax.legend(fontsize=7)
        ax.set_title(f"{title}\n(n={n})", fontsize=9)
        ax.set_xlabel("Homophily (H)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    # Panel 0: overall
    overall_n = len(tags)
    _draw(axes_flat[0], null_overall, real_overall, "Overall network", overall_n)

    # Per-type panels
    for i, ct in enumerate(eligible, start=1):
        n_ct = type_counts[ct]
        _draw(
            axes_flat[i],
            null_per_type.get(ct, []),
            real_per_type.get(ct),
            ct,
            n_ct,
        )

    # Hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{pair_name.replace('__vs__', ' vs ')}\nCell-type tag permutation (weighted homophily)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair(
    pair_name: str,
    n_perm: int,
    rng: np.random.Generator,
) -> dict | None:
    out_dir = OUTPUT_ROOT / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        nodes, edges = load_pair(pair_name)
    except FileNotFoundError as exc:
        logging.error("%s: %s", pair_name, exc)
        return None

    adj, tags = build_weighted_adjacency(nodes, edges)
    n_labelled = len(tags)
    n_edges = len(edges)

    if n_labelled < 10:
        logging.warning("%s: only %d labelled nodes — skipping", pair_name, n_labelled)
        return None

    logging.info(
        "%s: %d labelled nodes, %d edges, %d permutations",
        pair_name, n_labelled, n_edges, n_perm,
    )

    # Real homophily
    real_overall, real_per_type = compute_homophily(adj, tags)

    # Null distribution
    null_overall, null_per_type = run_permutations(adj, tags, n_perm, rng)

    # --- Save overall result ---
    overall_row = {"pair_name": pair_name, "cell_type": "OVERALL", "n_nodes": n_labelled}
    if real_overall is not None and null_overall:
        overall_row.update(_stats(real_overall, null_overall))
    else:
        overall_row.update({"real_H": None, "null_mean": None, "null_std": None,
                             "z_score": None, "empirical_p": None})
    pd.DataFrame([overall_row]).to_csv(
        out_dir / f"{pair_name}_homophily_overall.csv", index=False
    )

    # --- Save per-type results ---
    from collections import Counter
    type_counts = Counter(tags.values())
    per_type_rows = []
    for ct, real_h in sorted(real_per_type.items()):
        n_ct = type_counts.get(ct, 0)
        row: dict = {"pair_name": pair_name, "cell_type": ct, "n_nodes": n_ct}
        null_ct = null_per_type.get(ct, [])
        if n_ct >= MIN_NODES_PER_TYPE and null_ct:
            row.update(_stats(real_h, null_ct))
        else:
            row.update({"real_H": real_h, "null_mean": None, "null_std": None,
                        "z_score": None, "empirical_p": None,
                        "note": "below min_nodes threshold"})
        per_type_rows.append(row)
    pd.DataFrame(per_type_rows).to_csv(
        out_dir / f"{pair_name}_homophily_per_celltype.csv", index=False
    )

    # --- Plot ---
    plot_null_distributions(
        pair_name=pair_name,
        real_overall=real_overall,
        null_overall=null_overall,
        real_per_type=real_per_type,
        null_per_type=null_per_type,
        tags=tags,
        min_nodes=MIN_NODES_PER_TYPE,
        out_path=out_dir / f"{pair_name}_null_distributions.png",
    )

    # --- Convergence plot ---
    plot_convergence(
        pair_name=pair_name,
        null_overall=null_overall,
        real_overall=real_overall,
        out_path=out_dir / f"{pair_name}_convergence.png",
    )

    sparse_note = "sparse_network_interpret_with_caution" if pair_name in SPARSE_PAIRS else ""
    return {
        "pair_name": pair_name,
        "n_labelled_nodes": n_labelled,
        "n_edges": n_edges,
        "n_permutations": n_perm,
        "overall_real_H": real_overall,
        "overall_null_mean": float(np.mean(null_overall)) if null_overall else None,
        "overall_null_std": float(np.std(null_overall, ddof=1)) if len(null_overall) > 1 else None,
        "overall_z_score": _stats(real_overall, null_overall)["z_score"] if (real_overall and null_overall) else None,
        "overall_empirical_p": _stats(real_overall, null_overall)["empirical_p"] if (real_overall and null_overall) else None,
        "note": sparse_note,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_perm: int, seed: int) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    summary_rows = []
    for pair_name in tqdm(PAIR_NAMES, desc="Pairs", unit="pair"):
        row = run_pair(pair_name, n_perm, rng)
        if row is not None:
            summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUTPUT_ROOT / "tag_permutation_summary.csv", index=False
        )
        logging.info("Summary written to %s", OUTPUT_ROOT / "tag_permutation_summary.csv")

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Network tag permutation test (weighted homophily)"
    )
    parser.add_argument(
        "--n-permutations", type=int, default=DEFAULT_N_PERM,
        help=f"Number of label shuffles (default: {DEFAULT_N_PERM})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.n_permutations, args.seed))
