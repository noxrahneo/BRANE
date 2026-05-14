#!/usr/bin/env python3
# flake8: noqa: E501
"""Single-condition permutation threshold estimation.

For each condition, permutes sample order independently per gene (gene-wise
sample shuffle) to break gene-gene co-expression, recomputes the Pearson
correlation matrix, and records the maximum |r| across all gene pairs in the
upper triangle. The threshold is the mean of the permutation maxima, matching
the FWER-controlling approach used in script 30 for the differential networks.

Unlike the differential pipeline, no HVG intersection is required: the full
3,000 HVGs of each condition are used directly.

Inputs:
    results/08_pre_correlation/per_condition/<cond>_pseudobulk_logcpm.h5ad
    results/09_correlation/pearson/<cond>/<cond>_genes.csv

Outputs per condition in results/12_single_condition_thresholds/<cond>/:
    <cond>_permutation_maxima.csv
    <cond>_permutation_threshold.json
    <cond>_threshold_convergence.png
Summary:
    results/12_single_condition_thresholds/single_condition_thresholds_summary.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from utils.h5ad_compat import read_h5ad_compat
from utils.network_utils import resolve_base, save_json

CONDITIONS = [
    "ER_tumor",
    "Normal",
    "HER2_tumor",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_BRCA1_tumor",
    "Triple_negative_tumor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-condition permutation threshold estimation")
    parser.add_argument("--expr-dir", default="results/08_pre_correlation/per_condition",
                        help="Directory with <cond>_pseudobulk_logcpm.h5ad files")
    parser.add_argument("--corr-dir", default="results/09_correlation/pearson",
                        help="Per-condition correlation output directory (for gene lists)")
    parser.add_argument("--output-dir", default="results/12_single_condition_thresholds",
                        help="Output directory for thresholds")
    parser.add_argument("--condition", action="append", default=[],
                        help="Condition name to process. Repeat for multiple. Default: all six.")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations (default 1000)")
    parser.add_argument("--seed", type=int, default=42000,
                        help="Base random seed, incremented per condition")
    return parser.parse_args()


def read_expr_for_genes(expr_file: Path, genes: np.ndarray) -> np.ndarray:
    adata = read_h5ad_compat(expr_file)
    var = adata.var_names.astype(str)
    idx_map = {g: i for i, g in enumerate(var)}
    idx = np.array([idx_map[g] for g in genes], dtype=int)
    x = adata.X
    if sparse.issparse(x):
        x = sparse.csr_matrix(x).toarray()
    return np.asarray(x, dtype=np.float64)[:, idx]


def permute_gene_wise_samples(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle sample order independently per gene, breaking gene-gene correlations."""
    out = np.array(x, copy=True)
    n_samples, n_genes = out.shape
    for j in range(n_genes):
        out[:, j] = out[rng.permutation(n_samples), j]
    return out


def corr_from_expr(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def plot_convergence(maxima: list[float], out_png: Path, condition: str) -> None:
    """Running mean of permutation max |r| — shows threshold convergence."""
    run = np.arange(1, len(maxima) + 1)
    running_mean = np.cumsum(maxima) / run
    final_threshold = running_mean[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    # Left: running mean convergence
    axes[0].plot(run, running_mean, linewidth=1.8, color="#2a9d8f")
    axes[0].axhline(final_threshold, linestyle="--", linewidth=1.0, color="#888888",
                    label=f"threshold = {final_threshold:.4f}")
    axes[0].set_xlabel("Permutation")
    axes[0].set_ylabel("Running mean of max |r|")
    axes[0].set_title(f"{condition}: threshold convergence")
    axes[0].legend(frameon=False)

    # Right: histogram of permutation maxima
    axes[1].hist(maxima, bins=40, color="#f4a261", edgecolor="none", alpha=0.85)
    axes[1].axvline(final_threshold, linestyle="--", linewidth=1.5, color="#333333",
                    label=f"mean = {final_threshold:.4f}")
    axes[1].set_xlabel("Permutation max |r|")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{condition}: null distribution of max |r|")
    axes[1].legend(frameon=False)

    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def run_condition(
    condition: str,
    expr_root: Path,
    corr_root: Path,
    out_root: Path,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    t0 = time.perf_counter()

    expr_file = expr_root / f"{condition}_pseudobulk_logcpm.h5ad"
    genes_file = corr_root / condition / f"{condition}_genes.csv"

    if not expr_file.exists():
        raise FileNotFoundError(f"Missing expression file: {expr_file}")
    if not genes_file.exists():
        raise FileNotFoundError(f"Missing gene list: {genes_file}")

    genes = pd.read_csv(genes_file)["gene"].astype(str).to_numpy()
    if genes.size < 2:
        raise ValueError(f"{condition}: fewer than 2 genes, cannot compute correlations")

    x = read_expr_for_genes(expr_file, genes)
    n_profiles, n_genes = x.shape
    ii, jj = np.triu_indices(n_genes, k=1)
    rng = np.random.default_rng(seed)

    maxima: list[float] = []
    rows: list[dict[str, object]] = []

    for i in range(1, n_permutations + 1):
        x_perm = permute_gene_wise_samples(x, rng)
        corr_perm = corr_from_expr(x_perm)
        max_abs_r = float(np.max(np.abs(corr_perm[ii, jj])))
        maxima.append(max_abs_r)
        rows.append({"condition": condition, "permutation": i, "max_abs_r": max_abs_r})
        if i % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{condition}] permutation {i}/{n_permutations} — {elapsed:.1f}s elapsed")

    threshold = float(np.mean(maxima))

    cond_out = out_root / condition
    cond_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(cond_out / f"{condition}_permutation_maxima.csv", index=False)
    plot_convergence(maxima, cond_out / f"{condition}_threshold_convergence.png", condition)

    elapsed_total = time.perf_counter() - t0
    result: dict[str, object] = {
        "condition": condition,
        "n_genes": int(n_genes),
        "n_profiles": int(n_profiles),
        "n_permutations": n_permutations,
        "seed": seed,
        "threshold_abs_r": threshold,
        "elapsed_seconds": round(elapsed_total, 2),
    }
    save_json(cond_out / f"{condition}_permutation_threshold.json", result)

    print(f"[{condition}] profiles={n_profiles} genes={n_genes} "
          f"threshold_abs_r={threshold:.4f} elapsed={elapsed_total:.1f}s")
    return result


def main() -> None:
    args = parse_args()
    expr_root = resolve_base(args.expr_dir)
    corr_root = resolve_base(args.corr_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = args.condition if args.condition else CONDITIONS

    t_total = time.perf_counter()
    summaries: list[dict[str, object]] = []
    for i, condition in enumerate(conditions):
        summaries.append(run_condition(
            condition=condition,
            expr_root=expr_root,
            corr_root=corr_root,
            out_root=out_root,
            n_permutations=args.n_permutations,
            seed=args.seed + i,
        ))

    pd.DataFrame(summaries).to_csv(out_root / "single_condition_thresholds_summary.csv", index=False)
    total_elapsed = time.perf_counter() - t_total
    print(f"Done. Total elapsed: {total_elapsed:.1f}s. Thresholds written to: {out_root}")


if __name__ == "__main__":
    main()
