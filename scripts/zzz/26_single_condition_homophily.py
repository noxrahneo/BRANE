"""
Script 26: Single-Condition Network Homophily Permutation Test.

Tests whether cell-type labels are non-randomly distributed across
the topology of each single-condition co-expression network.
The metric is weighted neighbourhood homophily using |r| as edge
weight. The null is constructed by 1,000 label-preserving
Fisher-Yates shuffles.

Inputs:
  results/13_single_condition_networks/<cond>/<cond>_edges.tsv
  results/20_node_annotation/03_output_with_lfc/*_tagged_with_lfc.csv
    (cell_type column used as node labels; union across all six pairs)

Outputs:
  results/26_single_condition_homophily/<cond>/
    summary.csv         — observed H, null mean/sd, Z, empirical p
    per_cell_type.csv   — per-type H, Z, p
    convergence.png     — running null mean and Z-score
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = BASE_DIR / "results"

SINGLE_NET_DIR = RESULTS_DIR / "13_single_condition_networks"
TAGGED_DIR = RESULTS_DIR / "20_node_annotation" / "03_output_with_lfc"
OUTPUT_DIR = RESULTS_DIR / "26_single_condition_homophily"

CONDITIONS = [
    "ER_tumor",
    "HER2_tumor",
    "Normal",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor",
    "Triple_negative_BRCA1_tumor",
]

N_PERMUTATIONS = 1_000
MIN_CELL_TYPE_NODES = 5
RNG_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell-type label loading
# ---------------------------------------------------------------------------

def build_master_cell_type_map() -> dict[str, str]:
    """
    Union of gene→cell_type from all six tagged_with_lfc.csv files.
    If a gene appears in multiple pairs with different labels, the most
    frequent label is used.
    """
    records: list[tuple[str, str]] = []
    for path in sorted(TAGGED_DIR.glob("*_tagged_with_lfc.csv")):
        df = pd.read_csv(path, usecols=["gene", "cell_type"])
        df = df.dropna(subset=["cell_type"])
        df = df[df["cell_type"].str.strip() != ""]
        records.extend(zip(df["gene"], df["cell_type"]))

    if not records:
        raise FileNotFoundError(f"No tagged files found in {TAGGED_DIR}")

    frame = pd.DataFrame(records, columns=["gene", "cell_type"])
    # majority vote per gene
    master = (
        frame.groupby("gene")["cell_type"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    log.info("Master cell-type map: %d genes with labels", len(master))
    return master


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    edges_path: Path,
    cell_type_map: dict[str, str],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """
    Returns:
      adj   — {gene: {neighbour: weight}} using abs_r as weight
      tags  — {gene: cell_type} for labelled nodes only
    """
    df = pd.read_csv(edges_path, sep="\t")
    adj: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        a, b, w = str(row["gene_a"]), str(row["gene_b"]), float(row["abs_r"])
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w

    tags = {g: t for g, t in cell_type_map.items() if g in adj}
    return adj, tags


# ---------------------------------------------------------------------------
# Homophily computation
# ---------------------------------------------------------------------------

def node_homophily(
    v: str,
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
) -> float | None:
    t = tags.get(v)
    if t is None:
        return None
    neighbours = adj.get(v, {})
    labelled = {u: w for u, w in neighbours.items() if u in tags}
    if not labelled:
        return None
    total_w = sum(labelled.values())
    if total_w == 0:
        return None
    same_w = sum(w for u, w in labelled.items() if tags[u] == t)
    return same_w / total_w


def network_homophily(
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
) -> tuple[float, dict[str, float]]:
    """Returns overall H and per-cell-type H_t."""
    per_type: dict[str, list[float]] = {}
    values: list[float] = []
    for v in tags:
        h = node_homophily(v, adj, tags)
        if h is None:
            continue
        values.append(h)
        per_type.setdefault(tags[v], []).append(h)

    overall = float(np.mean(values)) if values else 0.0
    per_type_mean = {t: float(np.mean(vs)) for t, vs in per_type.items()}
    return overall, per_type_mean


def permutation_null(
    adj: dict[str, dict[str, float]],
    tags: dict[str, str],
    n_perms: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[dict[str, float]]]:
    genes = list(tags.keys())
    labels = list(tags.values())
    null_H: list[float] = []
    null_per_type: list[dict[str, float]] = []
    for _ in range(n_perms):
        rng.shuffle(labels)
        shuffled = dict(zip(genes, labels))
        h, pt = network_homophily(adj, shuffled)
        null_H.append(h)
        null_per_type.append(pt)
    return null_H, null_per_type


# ---------------------------------------------------------------------------
# Per-cell-type permutation statistics
# ---------------------------------------------------------------------------

def per_type_stats(
    obs_pt: dict[str, float],
    null_pt_list: list[dict[str, float]],
    tags: dict[str, str],
    min_nodes: int,
) -> pd.DataFrame:
    type_counts = pd.Series(list(tags.values())).value_counts()
    rows = []
    for t, obs in obs_pt.items():
        n_nodes = type_counts.get(t, 0)
        null_vals = [d.get(t, np.nan) for d in null_pt_list]
        null_vals = [v for v in null_vals if not np.isnan(v)]
        if n_nodes < min_nodes or len(null_vals) < 10:
            rows.append({"cell_type": t, "obs_H": obs, "n_nodes": n_nodes,
                         "null_mean": np.nan, "null_sd": np.nan,
                         "Z": np.nan, "empirical_p": np.nan})
            continue
        null_arr = np.array(null_vals)
        sd = null_arr.std()
        z = (obs - null_arr.mean()) / sd if sd > 0 else np.nan
        p = float((null_arr >= obs).mean())
        rows.append({"cell_type": t, "obs_H": obs, "n_nodes": n_nodes,
                     "null_mean": null_arr.mean(), "null_sd": sd,
                     "Z": z, "empirical_p": p})
    return pd.DataFrame(rows).sort_values(
        "Z", ascending=False, na_position="last"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_null_distribution(
    null_H: list[float],
    obs_H: float,
    null_pt_list: list[dict[str, float]],
    obs_pt: dict[str, float],
    tags: dict[str, str],
    n_nodes_overall: int,
    out_path: Path,
) -> None:
    """Multi-panel: overall H + one panel per cell type."""
    cell_types = sorted(obs_pt.keys())
    type_counts = pd.Series(list(tags.values())).value_counts()

    n_panels = 1 + len(cell_types)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows)
    )
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    def _draw(ax: plt.Axes, null_vals: list[float], obs_val: float,
              title: str) -> None:
        arr = np.array(null_vals)
        ax.hist(
            arr, bins=40, color="#2a9d8f", alpha=0.75,
            edgecolor="white", linewidth=0.4,
        )
        ax.axvline(
            obs_val, color="#e76f51", lw=1.8, linestyle="--",
            label=f"obs H = {obs_val:.3f}",
        )
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("Permuted H", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    _draw(
        axes_flat[0], null_H, obs_H,
        f"Overall network (n={n_nodes_overall})",
    )

    for i, ct in enumerate(cell_types):
        ax = axes_flat[i + 1]
        null_vals = [d.get(ct, np.nan) for d in null_pt_list]
        null_vals = [v for v in null_vals if not np.isnan(v)]
        n_ct = int(type_counts.get(ct, 0))
        _draw(ax, null_vals, obs_pt[ct], f"{ct} (n={n_ct})")

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    null_H: list[float], obs_H: float, out_path: Path
) -> None:
    arr = np.array(null_H)
    running_mean = np.cumsum(arr) / np.arange(1, len(arr) + 1)
    running_sd = np.array([arr[:i + 1].std() for i in range(len(arr))])
    with np.errstate(invalid="ignore"):
        running_z = np.where(
            running_sd > 0,
            (obs_H - running_mean) / running_sd,
            np.nan,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(running_mean, color="#2a9d8f", lw=1.2)
    ax1.axhline(
        obs_H, color="#e76f51", lw=1.2, linestyle="--",
        label=f"Observed H = {obs_H:.4f}",
    )
    ax1.set_xlabel("Permutations")
    ax1.set_ylabel("Running null mean H")
    ax1.legend(fontsize=8)

    ax2.plot(running_z, color="#264653", lw=1.2)
    ax2.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax2.set_xlabel("Permutations")
    ax2.set_ylabel("Running Z-score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_condition(
    cond: str, cell_type_map: dict[str, str], rng: np.random.Generator
) -> None:
    net_dir = SINGLE_NET_DIR / cond
    edges_path = net_dir / f"{cond}_edges.tsv"
    if not edges_path.exists():
        log.warning("Edge file not found for %s — skipping", cond)
        return

    out_dir = OUTPUT_DIR / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== %s ===", cond)
    adj, tags = build_graph(edges_path, cell_type_map)
    n_nodes = len(adj)
    n_labelled = len(tags)
    log.info("  Nodes: %d total, %d labelled", n_nodes, n_labelled)

    if n_labelled < 10:
        log.warning(
            "  Too few labelled nodes (%d) — skipping permutation", n_labelled
        )
        return

    obs_H, obs_pt = network_homophily(adj, tags)
    log.info("  Observed H = %.4f", obs_H)

    null_H, null_pt = permutation_null(adj, tags, N_PERMUTATIONS, rng)
    null_arr = np.array(null_H)
    null_mean = null_arr.mean()
    null_sd = null_arr.std()
    Z = (obs_H - null_mean) / null_sd if null_sd > 0 else np.nan
    emp_p = float((null_arr >= obs_H).mean())

    log.info(
        "  Null mean = %.4f  SD = %.4f  Z = %.2f  p = %.3f",
        null_mean, null_sd, Z, emp_p,
    )

    summary = pd.DataFrame([{
        "condition": cond,
        "n_nodes": n_nodes,
        "n_labelled": n_labelled,
        "obs_H": obs_H,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "Z": Z,
        "empirical_p": emp_p,
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)

    pt_df = per_type_stats(obs_pt, null_pt, tags, MIN_CELL_TYPE_NODES)
    pt_df.to_csv(out_dir / "per_cell_type.csv", index=False)

    plot_null_distribution(
        null_H, obs_H, null_pt, obs_pt, tags, n_labelled,
        out_dir / "null_distribution.png",
    )
    plot_convergence(null_H, obs_H, out_dir / "convergence.png")
    log.info("  Outputs written to %s", out_dir)


def main() -> None:
    cell_type_map = build_master_cell_type_map()
    rng = np.random.default_rng(RNG_SEED)
    for cond in CONDITIONS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_condition(cond, cell_type_map, rng)

    # Aggregate summary across all conditions
    summaries = []
    for cond in CONDITIONS:
        f = OUTPUT_DIR / cond / "summary.csv"
        if f.exists():
            summaries.append(pd.read_csv(f))
    if summaries:
        agg = pd.concat(summaries, ignore_index=True)
        agg.to_csv(OUTPUT_DIR / "all_conditions_summary.csv", index=False)
        log.info("\nAll-conditions summary:\n%s", agg.to_string(index=False))


if __name__ == "__main__":
    main()
